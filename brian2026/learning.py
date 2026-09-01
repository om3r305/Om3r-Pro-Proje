from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, Sequence
import json
import math
import time

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .samples import SupervisedSample

MODEL_SCHEMA_VERSION = "brian.supervised-model.v1"
Partition = Literal["train", "validation", "test"]


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    model_type: str
    model_version: str
    dataset_hash: str
    code_version: str
    feature_schema: str
    training_fold: int
    hyperparameters: tuple[tuple[str, Any], ...]
    created_timestamp: float
    preprocessing: str = "median-impute+missing-indicators; train-only"
    calibration: str = "validation-only-platt"


@dataclass(frozen=True, slots=True)
class ProbabilityPrediction:
    down: float
    neutral: float
    up: float


class SupervisedBaseline:
    model_type = "base"

    def __init__(self, metadata: ModelMetadata, *, random_state: int = 0) -> None:
        self.metadata = metadata
        self.random_state = random_state
        self.feature_names: tuple[str, ...] = ()
        self.pipeline: Pipeline | None = None
        self.calibrator: PlattCalibrator | None = None
        self.fit_partition: str | None = None

    def _estimator(self):
        raise NotImplementedError

    def fit(self, samples: Sequence[SupervisedSample], *, partition: Partition = "train") -> "SupervisedBaseline":
        if partition != "train":
            raise ValueError("model and preprocessing may fit only on train")
        self.feature_names = tuple(sorted({name for row in samples for name, _ in row.features}))
        x = _matrix(samples, self.feature_names)
        y = np.asarray([row.label for row in samples], dtype=int)
        if len(set(y.tolist())) < 2:
            raise ValueError("training requires at least two classes")
        steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
        if self.model_type == "logistic_regression":
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", self._estimator()))
        self.pipeline = Pipeline(steps)
        self.pipeline.fit(x, y)
        self.fit_partition = partition
        return self

    def predict_probability(self, samples: Sequence[SupervisedSample]) -> tuple[ProbabilityPrediction, ...]:
        if self.pipeline is None:
            raise RuntimeError("model not fitted")
        raw = self.pipeline.predict_proba(_matrix(samples, self.feature_names))
        classes = self.pipeline.named_steps["model"].classes_
        mapped = np.zeros((len(raw), 3), dtype=float)
        index = {-1: 0, 0: 1, 1: 2}
        for source, label in enumerate(classes):
            mapped[:, index[int(label)]] = raw[:, source]
        if self.calibrator is not None:
            mapped = self.calibrator.transform(mapped)
        return tuple(ProbabilityPrediction(*map(float, row)) for row in mapped)

    def calibrate(self, validation: Sequence[SupervisedSample], *, partition: Partition = "validation") -> None:
        if partition != "validation":
            raise ValueError("calibration may fit only on validation")
        if self.pipeline is None:
            raise RuntimeError("model not fitted")
        raw = self.pipeline.predict_proba(_matrix(validation, self.feature_names))
        classes = self.pipeline.named_steps["model"].classes_
        mapped = np.zeros((len(raw), 3), dtype=float)
        for source, label in enumerate(classes):
            mapped[:, {-1: 0, 0: 1, 1: 2}[int(label)]] = raw[:, source]
        self.calibrator = PlattCalibrator().fit(mapped, np.asarray([r.label for r in validation]), partition=partition)

    def save(self, directory: str | Path, *, metrics: dict[str, float] | None = None) -> Path:
        if self.pipeline is None:
            raise RuntimeError("model not fitted")
        payload = {"metadata": asdict(self.metadata), "features": self.feature_names,
                   "pipeline": self.pipeline, "calibrator": self.calibrator,
                   "model_type": self.model_type, "random_state": self.random_state,
                   "metrics": dict(metrics or {})}
        identity = sha256(json.dumps(asdict(self.metadata), sort_keys=True, default=str,
                                        separators=(",", ":")).encode()).hexdigest()
        path = Path(directory) / f"{identity}.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"immutable model artifact exists: {path}")
        joblib.dump(payload, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SupervisedBaseline":
        payload = joblib.load(path)
        target = LogisticRegressionBaseline if payload["model_type"] == "logistic_regression" else GradientBoostingBaseline
        obj = target(ModelMetadata(**payload["metadata"]), random_state=payload["random_state"])
        obj.feature_names, obj.pipeline, obj.calibrator = tuple(payload["features"]), payload["pipeline"], payload["calibrator"]
        obj.fit_partition = "train"
        return obj


class LogisticRegressionBaseline(SupervisedBaseline):
    model_type = "logistic_regression"
    def _estimator(self):
        return LogisticRegression(max_iter=1000, random_state=self.random_state)


class GradientBoostingBaseline(SupervisedBaseline):
    model_type = "gradient_boosting"
    def _estimator(self):
        allowed = {"n_estimators", "learning_rate", "max_depth", "subsample"}
        configured = {key: value for key, value in self.metadata.hyperparameters if key in allowed}
        return GradientBoostingClassifier(random_state=self.random_state, **configured)


class PlattCalibrator:
    def __init__(self) -> None:
        self.models: list[LogisticRegression | None] = []
        self.fit_partition: str | None = None

    def fit(self, probabilities: np.ndarray, labels: np.ndarray, *, partition: Partition) -> "PlattCalibrator":
        if partition != "validation":
            raise ValueError("calibrator may fit only on validation")
        self.models = []
        mapped = np.asarray([{-1: 0, 0: 1, 1: 2}[int(label)] for label in labels])
        for index in range(3):
            binary = (mapped == index).astype(int)
            if len(set(binary.tolist())) < 2:
                self.models.append(None)
            else:
                logits = np.log(np.clip(probabilities[:, index], 1e-9, 1 - 1e-9) /
                                np.clip(1 - probabilities[:, index], 1e-9, 1))[:, None]
                self.models.append(LogisticRegression(random_state=0).fit(logits, binary))
        self.fit_partition = partition
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        calibrated = np.zeros_like(probabilities)
        for index, model in enumerate(self.models):
            if model is None:
                calibrated[:, index] = probabilities[:, index]
            else:
                p = np.clip(probabilities[:, index], 1e-9, 1 - 1e-9)
                calibrated[:, index] = model.predict_proba(np.log(p / (1 - p))[:, None])[:, 1]
        totals = calibrated.sum(axis=1, keepdims=True)
        return calibrated / np.where(totals == 0, 1, totals)


def _matrix(samples: Sequence[SupervisedSample], feature_names: Sequence[str]) -> np.ndarray:
    return np.asarray([[dict(row.features).get(name, np.nan) if dict(row.features).get(name) is not None else np.nan
                        for name in feature_names] for row in samples], dtype=float)


def metadata_for(model_type: str, dataset_hash: str, code_version: str,
                 feature_schema: str, fold: int, hyperparameters: dict[str, Any] | None = None) -> ModelMetadata:
    return ModelMetadata(model_type, MODEL_SCHEMA_VERSION, dataset_hash, code_version,
                         feature_schema, fold, tuple(sorted((hyperparameters or {}).items())), time.time())
