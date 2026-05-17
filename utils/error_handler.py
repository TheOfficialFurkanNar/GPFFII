"""Error handling utilities for GPFF II framework"""
import traceback
from .logger import StructuredLogger
from .exceptions import GPFFIIError, PhysicalParameterError


class ErrorHandler:
    """Centralized error handling with logging integration"""

    def __init__(self, logger: StructuredLogger = None):
        self.logger = logger or StructuredLogger(device_id="gpffii", min_level="INFO")

    def handle(self, error: Exception, context: dict = None):
        """Handle and log an error with context"""
        error_data = {"type": type(error).__name__, "message": str(error)}
        if context:
            error_data["context"] = context

        self.logger.error(
            event="error_occurred",
            data=error_data,
            error=traceback.format_exc()
        )

        # Re-raise GPFF II specific errors; let others bubble up
        if isinstance(error, GPFFIIError):
            raise error

    def safe_execute(self, func, *args, **kwargs):
        """Execute a function with error handling, returning None on failure"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.handle(e, context={"function": func.__name__})
            return None

    def validate_parameters(self, params: dict, schema: dict):
        """Validate parameters against a schema of expected types and bounds"""
        for param, value in params.items():
            if param not in schema:
                raise PhysicalParameterError(f"Unknown parameter: {param}")

            expected_type = schema[param].get("type")
            if expected_type and not isinstance(value, expected_type):
                raise PhysicalParameterError(
                    f"Parameter {param} must be {expected_type}, got {type(value)}"
                )

            if "min" in schema[param] and value < schema[param]["min"]:
                raise PhysicalParameterError(f"Parameter {param} below minimum: {value}")

            if "max" in schema[param] and value > schema[param]["max"]:
                raise PhysicalParameterError(f"Parameter {param} above maximum: {value}")