"""
Custom Exception Classes for Pearls AQI Predictor.
"""

class AQIException(Exception):
    """Base exception class for Pearls AQI Predictor application."""
    def __init__(self, message: str = "An AQI processing error occurred", details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self):
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class DataPipelineError(AQIException):
    """Raised when data ingestion, cleaning, or feature engineering fails."""
    pass


class ModelNotFoundError(AQIException):
    """Raised when a requested machine learning model artifact cannot be found or loaded."""
    pass


class APIIntegrationError(AQIException):
    """Raised when an external API (e.g. OpenWeatherMap) call fails or returns invalid responses."""
    pass


class InvalidInputError(AQIException):
    """Raised when provided input parameters fail validation checks."""
    pass
