"""Custom exceptions for Pearls AQI Predictor."""


class PearlsAQIError(Exception):
    """Base exception class for Pearls AQI Predictor."""
    pass


class DataValidationError(PearlsAQIError):
    """Raised when data fails schema or range validation."""
    pass


class ProviderAPIError(PearlsAQIError):
    """Raised when an external API provider call fails."""
    pass


class FeatureEngineeringError(PearlsAQIError):
    """Raised when feature engineering or target generation fails."""
    pass


class DataLeakageError(PearlsAQIError):
    """Raised when potential data leakage is detected."""
    pass


class ModelTrainingError(PearlsAQIError):
    """Raised when model training or evaluation fails."""
    pass


class FeatureStoreError(PearlsAQIError):
    """Raised when interaction with Feature Store fails."""
    pass
