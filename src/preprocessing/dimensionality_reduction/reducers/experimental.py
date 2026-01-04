"""Scaffolding for experimental dimensionality reduction strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Sequence, Tuple

import importlib.util

import numpy as np

from .base import DimensionalityReducer
from ..registry import register_reducer


class ExperimentalReducer(DimensionalityReducer, ABC):
    """Base class for reducers that rely on experimental backends.

    Experimental reducers often depend on optional third-party packages or
    prototype implementations that may not be available in every environment.
    This base class keeps the wiring consistent with the production reducers
    while providing a small amount of ergonomics for handling optional
    dependencies and custom backend factories.
    """

    #: Experimental reducers should advertise their optional dependencies so
    #: orchestration layers can provide friendly error messages.
    required_packages: Sequence[str] = ()

    def __init__(
        self,
        *,
        n_components: int,
        random_state: Optional[int],
        backend_factory: Optional[Callable[[], object]] = None,
        fail_on_missing_dependencies: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(n_components=n_components, random_state=random_state, **kwargs)
        self._backend_factory = backend_factory
        self._dependency_error: Optional[ModuleNotFoundError] = None
        self._validate_dependencies(fail_on_missing_dependencies=fail_on_missing_dependencies)

    # ------------------------------------------------------------------
    # Dependency management
    # ------------------------------------------------------------------

    def _validate_dependencies(self, *, fail_on_missing_dependencies: bool) -> None:
        missing = self._missing_packages()
        if not missing:
            return

        message = (
            f"{self.method} reducer requires optional dependencies: {', '.join(missing)}. "
            "Install them and retry, or provide a pre-configured backend via"
            " the `backend_factory` argument."
        )
        self._dependency_error = ModuleNotFoundError(message)
        if fail_on_missing_dependencies:
            raise self._dependency_error

    @classmethod
    def _missing_packages(cls) -> Tuple[str, ...]:
        missing = []
        for package in cls.required_packages:
            if importlib.util.find_spec(package) is None:
                missing.append(package)
        return tuple(missing)

    # ------------------------------------------------------------------
    # Backend resolution hooks
    # ------------------------------------------------------------------

    def _resolve_backend(self) -> object:
        if self._dependency_error is not None:
            raise RuntimeError("Experimental reducer backend is unavailable") from self._dependency_error

        factory = self._backend_factory or self.create_default_backend
        backend = factory()
        if backend is None:
            raise RuntimeError(
                "Experimental reducer did not provide a backend. Either override"
                " `create_default_backend` to return a configured backend or pass"
                " a `backend_factory` when constructing the reducer."
            )
        return backend

    def create_default_backend(self) -> object:
        """Return a configured backend instance.

        Subclasses may override this to perform lazy imports or to configure
        their backend using parameters captured in ``self.extra_params``.
        Implementations should return the backend object that will be passed to
        :meth:`_fit_with_backend`.
        """

        raise RuntimeError(
            "No default backend factory supplied. Pass `backend_factory` when"
            " constructing the reducer or override `create_default_backend`."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, features: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
        backend = self._resolve_backend()
        return self._fit_with_backend(features, backend)

    @abstractmethod
    def _fit_with_backend(
        self,
        features: np.ndarray,
        backend: object,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        """Implement the algorithm using the provided backend."""


@register_reducer("msptsne")
class MSPTSNEReducer(ExperimentalReducer):
    """Placeholder scaffold for a multi-scale parametric t-SNE reducer.

    The implementation is intentionally left open-ended so future experiments
    can plug in custom backends (e.g. `multiscale_parametric_tsne`) without
    rewriting the surrounding orchestration. Typical usage will either provide
    a backend factory when instantiating the reducer or override
    :meth:`create_default_backend` and :meth:`_fit_with_backend` in a subclass.
    """

    method = "msptsne"
    required_packages: Sequence[str] = ("multiscale_parametric_tsne",)

    def _fit_with_backend(
        self,
        features: np.ndarray,
        backend: object,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        raise NotImplementedError(
            "MSPTSNEReducer is a scaffold. Provide an implementation by passing"
            " a backend_factory when constructing the reducer or by subclassing"
            " and overriding `_fit_with_backend`."
        )


__all__ = ["ExperimentalReducer", "MSPTSNEReducer"]
