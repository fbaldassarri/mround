# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""The recording stand-in that makes activation capture possible.

These exist because of a real failure. The first version of the stand-in was an
empty module that recorded its call and unwound the forward, which seemed
sufficient until it met ``mlx-lm``'s own Llama loop::

    mask = swa_mask if layer.use_sliding else fa_mask
    h = layer(h, mask, cache[i])

The loop reads an attribute off the block *before* calling it, so an empty
stand-in raises ``AttributeError`` on the first line and never records anything.
The stand-in now keeps the block it displaced and answers for it.

What these tests prove and do not prove. The stand-in is built against a local
reproduction of ``mlx.nn.Module``'s attribute semantics, transcribed from that
class: a dictionary subclass whose ``__setattr__`` routes containers into the
dictionary and everything else into ``__dict__``, and whose ``__getattr__``
consults the dictionary first. That is enough to exercise the delegation, the
recursion hazard in it, and the window during construction before there is
anything to delegate to. It is not enough to prove the stand-in works inside a
real model, because a wrong reproduction would fail here in the same direction it
fails there. `scripts/probe_blocks.py` is what settles that, on hardware.
"""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING, Any

import pytest

from mround.pipeline import blocks

if TYPE_CHECKING:
    from collections.abc import Iterator


class FakeModule(dict):  # type: ignore[type-arg]
    """A local reproduction of ``mlx.nn.Module``'s attribute behavior.

    Transcribed from that class rather than approximated, because the two quirks
    that matter are easy to get subtly wrong: a Module is itself a dictionary, so
    assigning one lands in the parent's dictionary rather than its ``__dict__``,
    and the assignment path calls ``hasattr`` first, which is what turns a naive
    delegating ``__getattr__`` into infinite recursion.
    """

    def __setattr__(self, key: str, value: Any) -> None:
        """Route containers into the dictionary, everything else into __dict__."""
        if isinstance(value, dict | list | tuple):
            if hasattr(self, key) and key not in self:
                delattr(self, key)
            self[key] = value
        else:
            super().__setattr__(key, value)
            self.pop(key, None)

    def __getattr__(self, key: str) -> Any:  # noqa: RET503
        """Consult the dictionary, then fail as an ordinary attribute would.

        The missing return is not an oversight. MLX's own version calls
        ``__getattribute__`` for its side effect of raising and falls off the
        end, and reproducing it exactly is the entire point of this class.
        """
        if key in self:
            return self[key]
        super().__getattribute__(key)


class FakeBlock(FakeModule):
    """A block carrying the kind of flag a model's own loop branches on.

    Deliberately not callable. The stand-in records and unwinds rather than
    delegating the call, so a block that could be called would let a regression
    in that direction pass unnoticed.
    """

    def __init__(self, *, use_sliding: bool = False) -> None:
        super().__init__()
        self.use_sliding = use_sliding
        self.name = "block"


@pytest.fixture
def recorder_type() -> Iterator[type]:
    """Build the stand-in against the local reproduction instead of MLX."""
    fake_nn = types.ModuleType("mlx.nn")
    fake_nn.Module = FakeModule  # type: ignore[attr-defined]
    fake_mlx = types.ModuleType("mlx")
    fake_mlx.nn = fake_nn  # type: ignore[attr-defined]

    saved = {name: sys.modules.get(name) for name in ("mlx", "mlx.nn")}
    sys.modules["mlx"] = fake_mlx
    sys.modules["mlx.nn"] = fake_nn
    blocks._recorder_type.cache_clear()
    try:
        yield blocks._recorder_type()
    finally:
        # Cleared on the way out too, so a machine that has real MLX does not
        # inherit a stand-in class built against the reproduction.
        blocks._recorder_type.cache_clear()
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class TestDelegation:
    def test_it_answers_for_the_block_it_displaced(self, recorder_type: type) -> None:
        # The failure that motivated all of this. mlx-lm reads `use_sliding` off
        # the layer before calling it, so a stand-in that cannot answer never
        # gets as far as recording anything.
        recorder = recorder_type(FakeBlock(use_sliding=True))
        assert recorder.use_sliding is True

    def test_it_answers_with_the_displaced_block_s_value_not_a_default(
        self, recorder_type: type
    ) -> None:
        assert recorder_type(FakeBlock(use_sliding=False)).use_sliding is False

    def test_an_attribute_neither_has_raises_attribute_error(self, recorder_type: type) -> None:
        # Not KeyError, and above all not RecursionError. A delegating
        # __getattr__ that reaches for the wrapped block through attribute
        # access rather than the dictionary recurses until the stack ends, and
        # the traceback for that says nothing about what went wrong.
        recorder = recorder_type(FakeBlock())
        with pytest.raises(AttributeError, match="no_such_attribute"):
            _ = recorder.no_such_attribute

    def test_construction_survives_having_nothing_to_delegate_to_yet(
        self, recorder_type: type
    ) -> None:
        # Module.__setattr__ calls hasattr before storing, which reaches
        # __getattr__ while the stand-in is still empty. Constructing at all is
        # the assertion.
        assert recorder_type(FakeBlock()) is not None


class TestRecording:
    def test_calling_it_unwinds_with_the_arguments(self, recorder_type: type) -> None:
        recorder = recorder_type(FakeBlock())
        with pytest.raises(blocks._CapturedCall) as caught:
            recorder("hidden", "mask", cache=None)
        assert caught.value.call_args == ("hidden", "mask")
        assert caught.value.call_kwargs == {"cache": None}

    def test_it_never_returns(self, recorder_type: type) -> None:
        # The stand-in cannot invent a return value: the shape of a block's
        # output is exactly what a generic stand-in does not know. Unwinding is
        # the only honest option, and it is also what makes capture cheap.
        recorder = recorder_type(FakeBlock())
        with pytest.raises(blocks._CapturedCall):
            recorder("hidden")

    def test_the_recorded_call_does_not_collide_with_exception_args(self) -> None:
        # BaseException.args is a real attribute. Storing the captured
        # positional arguments there would work by accident and break the first
        # time something inspected the exception.
        captured = blocks._CapturedCall(("hidden", "mask"), {"cache": None})
        assert captured.call_args == ("hidden", "mask")
        assert isinstance(captured.args, tuple)
        assert captured.args != captured.call_args

    def test_it_reads_as_control_flow_rather_than_a_failure(self) -> None:
        # It escapes into a traceback whenever something upstream of the block
        # raises first, and the message is the only context a reader gets.
        assert "not a failure" in str(blocks._CapturedCall((), {}))
