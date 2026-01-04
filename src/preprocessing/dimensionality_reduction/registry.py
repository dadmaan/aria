"""Reducer registry utilities for dimensionality reduction preprocessing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, Type

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .reducers.base import DimensionalityReducer

ReducerFactory = Callable[..., "DimensionalityReducer"]


class DimensionalityReducerRegistry:
    """Simple registry that maps method names to reducer factories."""

    def __init__(self) -> None:
        self._registry: Dict[str, ReducerFactory] = {}

    def register(self, method: str, factory: ReducerFactory, *, overwrite: bool = False) -> None:
        key = method.lower()
        if not overwrite and key in self._registry:
            raise ValueError(f"Reducer already registered for method '{method}'.")
        self._registry[key] = factory

    def create(self, method: str, **kwargs) -> "DimensionalityReducer":
        key = method.lower()
        try:
            factory = self._registry[key]
        except KeyError as exc:  # pragma: no cover - defensive; exercised via tests
            raise KeyError(f"No reducer registered for method '{method}'.") from exc
        return factory(**kwargs)

    def available_methods(self) -> Dict[str, ReducerFactory]:
        return dict(self._registry)


global_reducer_registry = DimensionalityReducerRegistry()


def register_reducer(
    method: str,
) -> Callable[[Type["DimensionalityReducer"]], Type["DimensionalityReducer"]]:
    """Class decorator to register reducers via their ``method`` name."""

    def decorator(cls: Type[DimensionalityReducer]) -> Type[DimensionalityReducer]:
        global_reducer_registry.register(method, cls)
        return cls

    return decorator


__all__ = [
    "DimensionalityReducerRegistry",
    "ReducerFactory",
    "global_reducer_registry",
    "register_reducer",
]
