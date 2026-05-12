"""Custom exceptions for GPFF II framework"""

class GPFFIIError(Exception):
    """Base exception for GPFF II errors"""
    pass

class ModeValidationError(GPFFIIError):
    """Raised when mode parameters are invalid"""
    pass

class NumericalInstabilityError(GPFFIIError):
    """Raised when numerical calculations become unstable"""
    pass

class PhysicalParameterError(GPFFIIError):
    """Raised when physical parameters are out of valid range"""
    pass

class ConfigurationError(GPFFIIError):
    """Raised when configuration is invalid"""
    pass

class CalculationError(GPFFIIError):
    """Raised when physics calculation fails"""
    pass
