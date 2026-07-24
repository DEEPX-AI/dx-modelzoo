class ConfigError(Exception):
    """Configuration value resolution failure."""

    pass


class YamlValidationError(Exception):
    """YAML config file validation failure."""

    pass


class ModelBuildError(Exception):
    """Model assembly failure."""

    pass
