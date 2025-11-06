"""

Module to explore the effect of time-dependence on quantities such as gs, Ca and K

"""


from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import pyvista 

import ufl
from dolfinx import mesh, fem, plot
from dolfinx.fem.petsc import create_vector, assemble_vector, assemble_matrix
from typing import Callable


def verify_convergence_to_known_steady_state(params: list[float], delta: Callable, kappa: Callable, domain: mesh.Mesh | None = None, domain_resolution: int = 100, animate: bool = False):
    pass


def solve_atmospheric_oscillations(params: list[float], amplitude: float, frequency: float, domain: mesh.Mesh | None = None, domain_resolution: int = 100, animate: bool = False):
    pass


def solve_stomatal_oscillations(params: list[float], amplitude: float, frequency: float, domain: mesh.Mesh | None = None, domain_resolution: int = 100, animate: bool = False):
    pass


def solve_absorption_oscillations(params: list[float], amplitude: float, frequency: float, domain: mesh.Mesh | None = None, domain_resolution: int = 100, animate: bool = False):
    pass

# simplified with a class based approach?

class TemporalSolver:
    def __init__(self, 
                 params: list[float], 
                 timing: tuple[float, float, float] = (0.0, 1.0, 0.01),
                 domain: mesh.Mesh | None = None, 
                 domain_resolution: int = 100, 
                 animate: bool = False,
                 update_absorption: Callable | None = None,
                 update_boundary_conditions: Callable | None = None,
                 initial_condition: Callable | None = None):
        self.tau = params[0]
        self.gamma = params[1]
        self.chi_ = params[2]
        self.t0 = timing[0]
        self.t_end = timing[1]
        self.dt = timing[2]
        self.domain = domain
        self.domain_resolution = domain_resolution
        self.animate = animate
        self.update_absorption = update_absorption
        self.update_boundary_conditions = update_boundary_conditions
        self.initial_condition = initial_condition

    def setup_domain(self):
        ...
    
    def setup_problem(self):
        ...

    def solve(self):
        ...