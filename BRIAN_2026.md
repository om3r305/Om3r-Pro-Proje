# BRIAN 2026

This repository now contains the first additive modernization layer under
`brian2026/`.  The legacy trader is intentionally left unchanged during the
foundation phase.

**Operating rule:** observe → decide → remember → replay → evaluate → promote.
No candidate can rewrite or promote itself into live-money execution without
passing the promotion gate and an explicit integration step.

See `brian2026/README.md` for architecture and `brian_shadow_demo.py` for a
no-order smoke demo.
