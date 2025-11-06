""" 

Module for utilities around steady state solver.

"""

import numpy as np
from mpi4py import MPI
import adios4dolfinx as a4x
import ufl 
from dolfinx import fem, mesh


def load_file(filename: str):
    """ Load the mesh and solution from file """
    mesh = a4x.read_mesh(filename, MPI.COMM_WORLD)
    V = fem.functionspace(mesh, ("CG", 1))
    uh = fem.Function(V)
    a4x.read_function(filename, uh, name="uh")
    return mesh, uh


def extract_solution_from_objects(mesh: mesh.Mesh, uh: fem.Function):
    """ Extract solution as numpy array from mesh and function objects """
    mesh.topology.create_connectivity(0, 1)
    V = fem.functionspace(mesh, ("CG", 1))
    imap = mesh.topology.index_map(0)
    nloc_vertices = imap.size_local 
    local_vertices = np.arange(nloc_vertices)

    v_dofs = fem.locate_dofs_topological(V, 0, local_vertices)
    xv = mesh.geometry.x[:nloc_vertices, 0].copy()
    uv = uh.x.array[v_dofs].copy()

    seq = np.argsort(xv)
    return xv[seq], uv[seq]


def extract_solution_from_file(filename: str):
    """ Extract the solution from the steady solver """
    mesh, uh = load_file(filename)
    return extract_solution_from_objects(mesh, uh)


def get_homogeneous_chii(params: list[float]):
    """ calculate chi(0) or C_i/C_a """
    tau, gamma, chi_ = params
    return chi_ + (1-chi_)/(1 + (tau/gamma)*np.tanh(tau))




# Utility functions for defining spatially varying parameters within UFL <------------------

def step_down(x, min, max, offset, epsilon=0.01):
    """ Step down function from max to min at offset """
    return min + (max - min) * 0.5 * (1 - ufl.tanh((x[0] - offset) / epsilon))


def step_up(x, min, max, offset, epsilon=0.01):
    """ Step up function from min to max at offset """
    return min + (max - min) * 0.5 * (1 + ufl.tanh((x[0] - offset) / epsilon))


def exp_down(x, beta):
    """ Exponential decay function """
    return beta*ufl.exp(-beta * x[0]) / (1 - np.exp(-beta))


def exp_up(x, beta):
    """ Exponential growth function """
    return beta*ufl.exp(beta * (x[0])) / (np.exp(beta) - 1)