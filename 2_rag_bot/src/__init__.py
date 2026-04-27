"""
src/__init__.py
~~~~~~~~~~~~~~~
Marks this directory as a Python package.
Exposes the primary public API surface for the career bot.
"""
from .agent import CareerAgent
from .ui import build_ui

__all__ = ["CareerAgent", "build_ui"]
