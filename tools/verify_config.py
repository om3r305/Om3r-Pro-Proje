# -*- coding: utf-8 -*-
from __future__ import annotations
import json, sys
from Proje1.core.config_loader import load_config
from Proje1.core.guardrails import get as cfg_get

def main():
    cfg = load_config()
    problems = []

    # Required tops
    for k in ["portfolio","entry_frac","risk","evo","drift"]:
        if k not in cfg:
            problems.append(f"missing key: {k}")
    # sanity checks
    por = cfg.get("portfolio", {})
    ef  = cfg.get("entry_frac", {})
    for k in ["dip","pred","news","ob"]:
        if k not in por: problems.append(f"portfolio.{k} missing")
        if k not in ef:  problems.append(f"entry_frac.{k} missing")

    lo, hi = float(cfg_get(cfg, "learning.autopatch.lo", 0.45)), float(cfg_get(cfg, "learning.autopatch.hi", 0.80))
    veto = float(cfg_get(cfg, "brain.veto_conf_min", 0.55))
    if not (lo <= veto <= hi):
        problems.append(f"brain.veto_conf_min out of bounds [{lo},{hi}]: {veto}")

    print(json.dumps({"ok": len(problems)==0, "problems": problems}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
