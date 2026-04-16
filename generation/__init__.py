"""
Purpose:

* Exposes answer generation utilities as a package.

Inputs:

* No direct runtime inputs in this file.

Outputs:

* Re-exported generator functions for easy imports.

Used in:

* Import paths that need generate_answer.
"""

from .generator import generate_answer

__all__ = ["generate_answer"]
