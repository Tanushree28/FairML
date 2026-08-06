# Legacy pipeline (frozen)

The original experiment scripts, kept for historical reference only. They
are NOT runnable from this directory (imports assume the repo root) and
contain two known training bugs fixed in `new_experiment/`:

1. Fairness terms computed via `.round().detach().numpy()` — no gradients,
   so only BCE trained the model and β had no effect.
2. Double sigmoid (in `forward` and again in the loss).

The COMPAS variant here also uses the leaking `duration = end - start`
feature (corr −0.78 with the label). The canonical pipeline for the paper
is `new_experiment/`; see `new_experiment/README.md`.

(The Streamlit dashboard was briefly archived here while it still read the
old top-level result paths. It has since been ported to the per-seed layout
and lives at `new_experiment/dashboard/`.)
