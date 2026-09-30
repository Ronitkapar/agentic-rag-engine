"""Make the repo root importable when pytest runs from any directory.

pytest prepends the *test file's* directory (tests/) to sys.path, not the repo
root, so `import app...` fails unless something puts the root there. Doing it
here means `pytest tests/`, `pytest`, and `python -m pytest` all behave the same
regardless of cwd.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
