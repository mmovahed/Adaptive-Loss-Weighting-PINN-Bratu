"""Backward-compatible entry point.

The benchmark no longer calls this comparator 'tuned' because no tuning is
performed inside a final Bratu case. The locked (1,3) pair is represented by
``bratu_fixed_selected.py`` and saved under method key ``fixed_selected``.
"""
from bratu_fixed_selected import main

if __name__ == "__main__":
    main()
