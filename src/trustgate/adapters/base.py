"""The SUTAdapter contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

from trustgate.models import Item, Output


class SUTAdapter(ABC):
    """A system under test. Given an item, produce an :class:`Output`."""

    #: Human-readable identity of this SUT (model + version + config), used in Run records.
    name: str

    @abstractmethod
    def predict(self, item: Item) -> Output:  # pragma: no cover - interface
        """Run the SUT on a single item and return its output."""
        raise NotImplementedError
