"""UCLIP toolkit package."""

from importlib import import_module

__all__ = [
    "core",
    "cli",
    "analysis",
    "viz",
]

# Lazily expose key submodules for convenience
for _name in __all__:
    import_module(f"uclip.{_name}")
