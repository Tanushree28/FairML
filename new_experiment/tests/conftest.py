import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent   # new_experiment/
ROOT = HERE.parent                              # repo root
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
