from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    fold: int
    train: range
    validation: range
    test: range
    purge: int
    embargo: int

    def to_dict(self) -> dict[str, object]:
        return {"fold": self.fold,
                "train": (self.train.start, self.train.stop),
                "validation": (self.validation.start, self.validation.stop),
                "test": (self.test.start, self.test.stop),
                "purge": self.purge, "embargo": self.embargo}
    def validate(self) -> None:
        sets = [set(self.train), set(self.validation), set(self.test)]
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise ValueError("walk-forward partitions overlap")
        if self.validation.start - self.train.stop < self.purge:
            raise ValueError("train/validation purge violated")
        if self.test.start - self.validation.stop < self.purge:
            raise ValueError("validation/test purge violated")


@dataclass(frozen=True, slots=True)
class WalkForwardSplitter:
    train_size: int
    validation_size: int
    test_size: int
    purge: int = 0
    embargo: int = 0
    max_folds: int | None = None

    def split(self, observations: int) -> tuple[WalkForwardFold, ...]:
        if min(self.train_size, self.validation_size, self.test_size) <= 0:
            raise ValueError("partition sizes must be positive")
        if self.purge < 0 or self.embargo < 0:
            raise ValueError("purge and embargo must be non-negative")
        folds: list[WalkForwardFold] = []
        train_start, train_stop, fold_number = 0, self.train_size, 0
        while True:
            val_start = train_stop + self.purge
            val_stop = val_start + self.validation_size
            test_start = val_stop + self.purge
            test_stop = test_start + self.test_size
            if test_stop > observations:
                break
            fold = WalkForwardFold(
                fold_number, range(train_start, train_stop),
                range(val_start, val_stop), range(test_start, test_stop),
                self.purge, self.embargo,
            )
            fold.validate()
            folds.append(fold)
            if self.max_folds is not None and len(folds) >= self.max_folds:
                break
            # Expanding train; the embargo separates the previous test from
            # data newly admitted to the next training boundary.
            train_stop = test_stop + self.embargo
            fold_number += 1
        return tuple(folds)
