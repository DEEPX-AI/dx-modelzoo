class DxmzError(Exception):
    """Base class for public dx-modelzoo domain failures."""


class ConfigError(DxmzError, KeyError):
    """Configuration value resolution failure."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


class YamlValidationError(DxmzError):
    """YAML config file validation failure."""

    pass


class ModelBuildError(DxmzError):
    """Model assembly failure."""

    pass


class SessionCreationError(DxmzError):
    """Session creation or artifact resolution failure."""


class DatasetValidationError(DxmzError, FileNotFoundError):
    """Dataset structure or integrity validation failure."""
