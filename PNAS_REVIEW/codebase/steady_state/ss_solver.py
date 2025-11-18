""" 

Module for solving steady state problems.

"""

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import adios4dolfinx as a4x
import ufl 
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from typing import Callable

def solver(params: list[float], delta: Callable, kappa: Callable, domain: mesh.Mesh | None = None, output_path: str = "../files/run.bp", save: bool = False, domain_resolution: int = 100):
    """ 
    Solve the steady-state model for given parameters, delta(z), kappa(z)
    
    Parameters
    ----------
    params : list[float]
        List of parameters [tau, gamma, chi_]
    delta : callable
        Function mapping spatial coordinate to delta(z)
    kappa : callable
        Function mapping spatial coordinate to kappa(z)
    domain: mesh.Mesh | None
        The mesh domain to solve on. If None, a default mesh will be used.
    output_path : str
        Path to save the output files
    save : bool
        Whether to save the solution to file
    """
    tau, gamma, chi_ = params
    if domain is None:
        domain = mesh.create_interval(MPI.COMM_WORLD, domain_resolution, [0.0, 1.0])
    V = fem.functionspace(domain, ("CG", 1))

    # Create facet tags 
    tdim = domain.topology.dim 
    fdim = tdim - 1
    facets_left = mesh.locate_entities(domain, fdim, lambda x: np.isclose(x[0], 0.0))
    facets_right = mesh.locate_entities(domain, fdim, lambda x: np.isclose(x[0], 1.0))
    facet_indices = np.concatenate([facets_left, facets_right])
    facet_tags = np.concatenate([np.full_like(facets_left, 1), np.full_like(facets_right, 2)])
    mt = mesh.meshtags(domain, fdim, facet_indices, facet_tags)
    dx = ufl.Measure("dx", domain=domain)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=mt)

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    tau_ = fem.Constant(domain, PETSc.ScalarType(tau))
    gamma_ = fem.Constant(domain, PETSc.ScalarType(gamma))
    chi_star = fem.Constant(domain, PETSc.ScalarType(chi_))

    # build forms
    a = - delta(x) * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx \
        - tau_**2 * kappa(x) * u * v * dx \
        - gamma_ * u * v * ds(1)
    
    L = - tau_**2 * kappa(x) * chi_star * v * dx \
        - gamma_ * v * ds(1)

    problem = LinearProblem(a, L, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    if save:
        # save solution
        a4x.write_mesh(output_path, domain)
        a4x.write_function(output_path, uh, name="CO2_profile")
    return domain, uh 

