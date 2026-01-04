"""Base classes and registry for backend adapter abstraction.

This module defines the core abstractions for backend adapters in the system:

- BaseAgentAdapter: Abstract base class defining the interface for RL framework adapters
- BackendRegistry: Singleton registry for managing and creating adapter instances

These classes provide a common interface for different machine learning backends,
enabling polymorphic behavior and easy extension with new frameworks.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional, Type

from ..interfaces.agents import GenerativeAgent, PerceivingAgent


class BaseAgentAdapter(metaclass=abc.ABCMeta):
    """Base class for RL framework adapters."""

    @abc.abstractmethod
    def create_generative_agent(
        self, config: Dict[str, Any], **kwargs: Any
    ) -> GenerativeAgent:
        """Create a generative agent instance.

        Args:
            config: Configuration dictionary
            **kwargs: Additional framework-specific arguments

        Returns:
            GenerativeAgent instance
        """

    @abc.abstractmethod
    def create_perceiving_agent(
        self, config: Dict[str, Any], **kwargs: Any
    ) -> PerceivingAgent:
        """Create a perceiving agent instance.

        Args:
            config: Configuration dictionary
            **kwargs: Additional framework-specific arguments

        Returns:
            PerceivingAgent instance
        """

    @abc.abstractmethod
    def get_backend_name(self) -> str:
        """Get the name of this backend.

        Returns:
            Backend identifier string
        """


class BackendRegistry:
    """Registry for managing backend adapters.

    This singleton registry maintains a mapping of backend names to their
    corresponding adapter classes. It provides methods for registering new
    backends, retrieving adapter classes, and instantiating adapters.

    The registry enables dynamic backend selection and supports the factory
    pattern for creating adapter instances with proper configuration.
    """

    _adapters: Dict[str, Type[BaseAgentAdapter]] = (
        {}
    )  # Class-level storage for registered adapters

    @classmethod
    def register(cls, name: str, adapter_class: Type[BaseAgentAdapter]) -> None:
        """Register a backend adapter.

        Args:
            name: Backend name (e.g., 'sb3')
            adapter_class: Adapter class
        """
        cls._adapters[name] = adapter_class  # Store adapter class by name

    @classmethod
    def get_adapter(cls, name: str) -> Type[BaseAgentAdapter]:
        """Get a registered adapter class.

        Args:
            name: Backend name

        Returns:
            Adapter class

        Raises:
            KeyError: If backend not registered
        """
        if name not in cls._adapters:  # Check if backend is registered
            raise KeyError(
                f"Backend '{name}' not registered. Available: {list(cls._adapters.keys())}"
            )
        return cls._adapters[name]  # Return the adapter class

    @classmethod
    def create_adapter(cls, name: str, *args: Any, **kwargs: Any) -> BaseAgentAdapter:
        """Create an adapter instance.

        Args:
            name: Backend name
            *args: Constructor arguments
            **kwargs: Constructor keyword arguments

        Returns:
            Adapter instance
        """
        adapter_class = cls.get_adapter(name)  # Retrieve adapter class
        return adapter_class(*args, **kwargs)  # Instantiate and return

    @classmethod
    def list_backends(cls) -> list[str]:
        """List all registered backends.

        Returns:
            List of backend names
        """
        return list(cls._adapters.keys())
