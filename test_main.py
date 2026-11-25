"""Run the whole test suite with one command: ``python test_main.py``.

Extra arguments are passed to pytest, e.g. ``python test_main.py -k features``.
"""

import sys

import pytest

if __name__ == "__main__":
    # "tests" = folder with all test files; -v lists every test by name
    sys.exit(pytest.main(["tests", "-v", *sys.argv[1:]]))
