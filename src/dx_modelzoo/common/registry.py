from __future__ import annotations

from typing import Any, Dict, List, Type, Union


def _derive_name(cls: type) -> str:
    """Derive a snake_case registry key from a class name.

    Strips common suffixes (``Dataset``) then converts CamelCase to snake_case.

    Examples::

        BSD68Dataset   → bsd68
        COCOPoseDataset → coco_pose
        OxfordPetDataset → oxford_pet
    """
    name = cls.__name__
    for suffix in ("Dataset",):
        if name.endswith(suffix) and len(name) > len(suffix):
            name = name[: -len(suffix)]
            break
    # s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    # s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
    # return s.lower()

    return name


class Registry:
    """Named registry mapping string keys to classes.

    Usage::

        REGISTRY = Registry("example")

        @REGISTRY.register("explicit_name")
        class Foo:
            ...

        @REGISTRY.register          # auto-derive: "bar" from BarDataset
        class BarDataset:
            ...

        cls = REGISTRY.get("explicit_name")
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._registry: Dict[str, Type[Any]] = {}

    def register(self, name_or_cls: Union[str, type, None] = None):
        """Decorator to register a class.

        Can be used as:
            ``@REGISTRY.register("name")``  — explicit name
            ``@REGISTRY.register``           — auto-derive from class name
        """
        if name_or_cls is None or isinstance(name_or_cls, str):
            # Called with parentheses: @register("name") or @register()
            explicit_name = name_or_cls

            def decorator(cls):
                key = explicit_name or _derive_name(cls)
                if key in self._registry:
                    raise ValueError(f"'{key}' already registered in {self.name} registry")
                self._registry[key] = cls
                cls.__registry_key__ = key  # Store registry key in class
                return cls

            return decorator
        else:
            # Called without parentheses: @register
            cls = name_or_cls
            key = _derive_name(cls)
            if key in self._registry:
                raise ValueError(f"'{key}' already registered in {self.name} registry")
            self._registry[key] = cls
            cls.__registry_key__ = key  # Store registry key in class
            return cls

    def get(self, name: str) -> Type[Any]:
        """Retrieve a registered class by name."""
        if name not in self._registry:
            raise KeyError(
                f"'{name}' not found in {self.name} registry. " f"Available: {sorted(self._registry.keys())}"
            )
        return self._registry[name]

    def list(self) -> List[str]:
        """List all registered names."""
        return list(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __len__(self) -> int:
        return len(self._registry)
