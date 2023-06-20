"""
Unit and regression test for the ligbinddiff package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import ligbinddiff


def test_ligbinddiff_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "ligbinddiff" in sys.modules
