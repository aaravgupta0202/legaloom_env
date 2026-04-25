"""
Make sure pytest can import project modules regardless of where it's invoked.
Adds the project root to sys.path before any test module is collected.
"""
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
