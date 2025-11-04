""" 

Module for CO2 solution given functions that map z -> kappa(z) and z-> delta(z)

"""

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import adios4dolfinx as a4x
import ufl 
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmshio
import matplotlib.pyplot as plt

def solver(tau, gamma, chi_star, delta=lambda x: 1.0, kappa=lambda x: 1.0, order=1, mesh_file="./files/1D_mesh.msh", filename="./files/run.bp"):
    """ 
    Solve the steady-state model for given tau, gamma, delta(z), kappa(z)
    
    Parameters
    ----------
    tau : float
        The value of tau in the model
    gamma : float
        The value of gamma in the model
    """
    master = 0
    mesh, cell_tags, facet_tags = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, master, gdim=1)
    MESOPHYL_TAG = 1
    STOMATAL_INTERFACE_TAG = 2

    # define function space
    V = fem.functionspace(mesh, ("Lagrange", order))
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    tau_ = fem.Constant(mesh, PETSc.ScalarType(tau))
    gamma_ = fem.Constant(mesh, PETSc.ScalarType(gamma))
    chi_ = fem.Constant(mesh, PETSc.ScalarType(chi_star))

    a = -1*delta(x) * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx(MESOPHYL_TAG) \
        - tau_**2 * kappa(x) * u * v * dx(MESOPHYL_TAG) \
        - gamma_ * u * v * ds(STOMATAL_INTERFACE_TAG)
    L = - tau_**2 * kappa(x) * chi_ * v * dx(MESOPHYL_TAG) \
        - gamma_ * v * ds(STOMATAL_INTERFACE_TAG)
    
    problem = LinearProblem(a, L, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve() 

    # save solution
    a4x.write_mesh(filename, mesh)
    a4x.write_meshtags(filename, mesh, cell_tags, meshtag_name="cell_tags")
    a4x.write_meshtags(filename, mesh, facet_tags, meshtag_name="facet_tags")
    a4x.write_function(filename, uh, name="CO2_profile")

def extract_solution(filename, order=1):
    """ Extract the solution from the steady solver """
    mesh = a4x.read_mesh(filename, MPI.COMM_WORLD)
    V = fem.functionspace(mesh, ("Lagrange", order))
    uh = fem.Function(V)
    a4x.read_function(filename, uh, name="CO2_profile")

    # plot the solution in matplotlib over z=0 to z=1
    mesh.topology.create_connectivity(0, 1)
    imap = mesh.topology.index_map(0)
    nloc_vertices = imap.size_local 
    local_vertices = np.arange(nloc_vertices)

    v_dofs = fem.locate_dofs_topological(V, 0, local_vertices)
    xv = mesh.geometry.x[:nloc_vertices, 0].copy()
    uv = uh.x.array[v_dofs].copy()

    seq = np.argsort(xv)
    return xv[seq], uv[seq]


def plotter(filename, resolution = 100):
    """ Plot the solution from the steady solver """
    z, c = extract_solution(filename, resolution=resolution)
    plt.plot(z, c)
    plt.xlabel("z")
    plt.ylabel("CO2 concentration")
    plt.title("Steady-state CO2 profile")
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.grid()
    plt.show()
    return z, c
 
def step_down(x, min, max, offset, epsilon=0.01):
    """ Step down function from max to min at offset """
    return min + (max - min) * 0.5 * (1 - ufl.tanh((x[0] - offset) / epsilon))

def step_up(x, min, max, offset, epsilon=0.01):
    """ Step up function from min to max at offset """
    return min + (max - min) * 0.5 * (1 + ufl.tanh((x[0] - offset) / epsilon))

def exp_down(x, beta):
    """ Exponential decay function """
    return ufl.exp(-beta * x[0]) / (1 - np.exp(-beta))

def exp_up(x, beta):
    """ Exponential growth function """
    return ufl.exp(beta * (x[0])) / (np.exp(beta) - 1)

