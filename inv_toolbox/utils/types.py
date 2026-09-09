# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

"""Provide utils to work with list."""

from collections.abc import Iterable, Sequence
from typing import TypeVar, Union, overload

import numpy as np
import numpy.typing as npt

NDArrayFloat = npt.NDArray[np.floating]
NDArrayInt = npt.NDArray[np.integer]
NDArrayBool = npt.NDArray[np.bool_]
Int = Union[int, NDArrayInt, Sequence[int]]

_Object = TypeVar("_Object", bound=object)


@overload
def object_or_object_sequence_to_list(_input: Iterable[_Object]) -> list[_Object]: ...
@overload
def object_or_object_sequence_to_list(_input: _Object) -> list[_Object]: ...
def object_or_object_sequence_to_list(
    _input: Union[_Object, Iterable[_Object]],
) -> list[_Object]:
    """
    Convert a singleton or an iterable of this object to a list of object.

    Parameters
    ----------
    _input : Union[_Object, Iterable[_Object]]
        Either a single object, or an iterable of objects.

    Returns
    -------
    List[_Object]
        ``[_input]`` if `_input` is not iterable (or is a `str`/`bytes`
        instance, which are treated as atomic values rather than sequences
        of characters/bytes), otherwise ``list(_input)``.
    """
    if isinstance(_input, (str, bytes)):
        return [_input]  # type: ignore
    if isinstance(_input, Iterable):
        return list(_input)
    return [_input]
