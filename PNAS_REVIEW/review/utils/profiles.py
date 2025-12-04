""" 

module of general profiles

"""

import numpy as np
from typing import Callable 
import functools


class StepProfile:
    def __init__(self, epsilon: float = 0.01, direction: str = "down"):
        self.minimum = None 
        self.maximum = None
        self.offset = None
        self.epsilon = epsilon
        self.direction = None 
        if direction == "down":
            self.direction = -1
        elif direction == "up":
            self.direction = 1
        else:
            raise ValueError("Direction must be 'down' or 'up'")
    
    def generalize(self) -> Callable[[np.ndarray, float], np.ndarray]:
        self.check_if_populated()
        """ Return a function that can be used in the general time-dependent solver """
        def profile(x: np.ndarray, t: float) -> np.ndarray:
            return self.minimum + (self.maximum - self.minimum) * 0.5 * (1 + self.direction * np.tanh((x - self.offset) / self.epsilon))
        return profile
    
    def steadify(self) -> Callable[[np.ndarray], np.ndarray]:
        self.check_if_populated()
        """ Return a function that can be used in the steady solver """
        return functools.partial(self.generalize(), t=0.0)
    
    def populate_limits(self, minimum: float, maximum: float, offset: float) -> None:
        self.minimum = minimum
        self.maximum = maximum
        self.offset = offset

    def populate_rho(self, rho_ratio: float, rho_lambda: float) -> None: 
        if self.direction == -1:
            self.maximum = 1/((1-rho_lambda) + rho_lambda*rho_ratio)
            self.minimum = rho_ratio*self.maximum
            self.offset = 1 - rho_lambda

        elif self.direction == 1:
            self.maximum = 1/((1-rho_lambda)*rho_ratio + rho_lambda)
            self.minimum = rho_ratio*self.maximum
            self.offset = 1 - rho_lambda

    def check_if_populated(self) -> None:
        if self.minimum is None or self.maximum is None or self.offset is None:
            raise ValueError("Populate parameters using .populate_limits() or .populate_rho() before calling")
    



class ExponentialProfile:
    def __init__(self, beta: float, direction: str = "down"):
        self.beta = beta
        self.direction = None 
        if direction == "down":
            self.direction = -1
        elif direction == "up":
            self.direction = 1
        else:
            raise ValueError("Direction must be 'down' or 'up'")

    def generalize(self) -> Callable[[np.ndarray, float], np.ndarray]:
        """ Return a function that can be used in the general time-dependent solver """
        def profile(x: np.ndarray, t: float) -> np.ndarray:
            return (self.beta*np.exp(self.direction*self.beta*x))/(self.direction*(np.exp(self.direction*self.beta) - 1))
        return profile
    
    def steadify(self) -> Callable[[np.ndarray], np.ndarray]:
        """ Return a function that can be used in the steady solver """
        return functools.partial(self.generalize(), t=0.0)




class OscillatorProfile:
    def __init__(self, amplitude: float, period: float):
        self.amplitude = amplitude
        self.period = period

    def generalize(self) -> Callable[[np.ndarray, float], np.ndarray]:
        """ Return a function that can be used in the general time-dependent solver """
        def profile(x: np.ndarray, t: float) -> np.ndarray:
            return 1.0 + self.amplitude * np.sin(2.0 * np.pi * t / self.period)
        return profile
    
