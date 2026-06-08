from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import BinaryIO, Optional


@dataclass(frozen=True)
class _ModulePrefixRemap:
    from_prefix: str
    to_prefix: str

    def apply(self, module: str) -> str:
        if module.startswith(self.from_prefix):
            return f"{self.to_prefix}{module[len(self.from_prefix):]}"
        return module


class _RemappingUnpickler(pickle.Unpickler):
    def __init__(self, file: BinaryIO, *, prefix_remaps: list[_ModulePrefixRemap]):
        super().__init__(file)
        self._prefix_remaps = prefix_remaps

    def find_class(self, module: str, name: str):
        for remap in self._prefix_remaps:
            module = remap.apply(module)
        return super().find_class(module, name)


def load_pickle_compat(path: str | Path):
    """
    Load a pickle file with a small compatibility shim.

    This mainly targets pickles created under NumPy 2.x (which reference
    internal modules under ``numpy._core``) being loaded in environments
    with NumPy 1.x (which uses ``numpy.core``).
    """
    path = Path(path)
    with path.open("rb") as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError as exc:
            missing = getattr(exc, "name", "") or ""
            if not missing.startswith("numpy._core"):
                raise

            f.seek(0)
            unpickler = _RemappingUnpickler(
                f, prefix_remaps=[_ModulePrefixRemap("numpy._core", "numpy.core")]
            )
            return unpickler.load()


def load_pickle_compat_or_raise(path: str | Path):
    """
    Same as ``load_pickle_compat``, but with a clearer error message on failure.
    """
    try:
        return load_pickle_compat(path)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Failed to load pickle {Path(path)!s}. Your environment is missing a module "
            f"required by the pickle ({getattr(exc, 'name', None)!r}). "
            "If this is a NumPy/sklearn model pickle, regenerate it in the current environment "
            "or align dependency versions."
        ) from exc

