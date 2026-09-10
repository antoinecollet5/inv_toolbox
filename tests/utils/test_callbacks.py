# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Antoine COLLET

"""Tests for the Callback class."""

from inv_toolbox.utils.callbacks import Callback


def test_callback_init() -> None:
    cb = Callback()
    assert cb.itercount() == 0


def test_callback_call_increments_counter() -> None:
    cb = Callback()
    cb()
    cb(1, 2, foo="bar")
    cb("anything")
    assert cb.itercount() == 3


def test_callback_clear() -> None:
    cb = Callback()
    cb()
    cb()
    assert cb.itercount() == 2
    cb.clear()
    assert cb.itercount() == 0
