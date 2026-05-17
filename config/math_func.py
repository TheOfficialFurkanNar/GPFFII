import math
import numpy as np
from config.constants import PI

# Partial derivatives

class PartialDerivative:
    def __init__(self):
        pass

    def partial_x(self, f, x, y, h=1e-5):
        """∂f/∂x at point (x, y)"""
        return (f(x + h, y) - f(x - h, y)) / (2 * h)

    def partial_y(self, f, x, y, h=1e-5):
        """∂f/∂y at point (x, y)"""
        return (f(x, y + h) - f(x, y - h)) / (2 * h)

    def gradient(self, f, x, y, h=1e-5):
        """∇f = (∂f/∂x, ∂f/∂y) — gradient vector at (x, y)"""
        return np.array([self.partial_x(f, x, y, h), self.partial_y(f, x, y, h)])







