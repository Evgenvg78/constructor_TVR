"""IO utilities for working with .tvr2 files."""

from .tvr_io import (
    TVRFile,
    TVRIOError,
    read_tvr2,
    write_tvr2,
    iter_tvr2_triplets,
)

__all__ = [
    "TVRFile",
    "TVRIOError",
    "read_tvr2",
    "write_tvr2",
    "iter_tvr2_triplets",
]
