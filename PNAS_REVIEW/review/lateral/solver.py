""" 

Module for solving a simple 3d continuous cylindrical problem wit discrete stomata

"""

from mpi4py import MPI
from petsc4py import PETSc
import adios4dolfinx as a4x
from pathlib import Path
import ufl 
from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmshio
from review.lateral.create_mesh import ASPECT_RATIO, FILENAME

# aspect ratio is on the order of 0.1

class Steady3DSolver:
    def __init__(self, 
                params: list[float], # tau, gamma, chi_
                stomatal_ratio: float = 1.0,
                stomatal_blur: float = 0.02,
                aspect_ratio: float = ASPECT_RATIO,
                meshfile: str | Path = FILENAME,
                filename: str | Path = "../files/3d/steady3d.bp",
                order: int = 1) -> None:
        self.tau = params[0]
        self.gamma = params[1]
        self.chi_ = params[2]
        self.stomatal_ratio = stomatal_ratio
        self.stomatal_blur = stomatal_blur
        self.aspect_ratio = aspect_ratio
        self.stomatal_radius = stomatal_ratio * aspect_ratio
        self.meshfile = meshfile if isinstance(meshfile, Path) else Path(meshfile)
        self.filename = filename if isinstance(filename, Path) else Path(filename)
        self.order = order

    def solve(self) -> None:
        mesh, cell_tags, facet_tags = gmshio.read_from_msh(self.meshfile, MPI.COMM_SELF, 0, gdim=3)
        #
        AIRSPACE_TAG = 1
        TOP_TAG = 2
        BOTTOM_TAG = 3
        CURVED_TAG = 4
        #
        functionspace = fem.functionspace(mesh, ("CG", self.order)) 
        dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)
        ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)

        tau2 = fem.Constant(mesh, default_scalar_type(self.tau**2))
        chi_ = fem.Constant(mesh, default_scalar_type(self.chi_))

        chi = ufl.TrialFunction(functionspace)
        v   = ufl.TestFunction(functionspace)
        x   = ufl.SpatialCoordinate(mesh)
        #
        phi   = (x[0]**2 + x[1]**2 - self.stomatal_radius**2) # type: ignore[reportIndexIssue]
        gamma = self.gamma * 0.5*(1 - ufl.tanh(phi/self.stomatal_blur))
        #
        # Weak form
        a = ufl.inner(ufl.grad(chi), ufl.grad(v)) * dx(AIRSPACE_TAG) \
            + tau2 * chi * v * dx(AIRSPACE_TAG) \
            + gamma * chi * v * ds(BOTTOM_TAG)
        
        L = tau2 * chi_ * v * dx(AIRSPACE_TAG) \
            + gamma * v * ds(BOTTOM_TAG) 
        
        problem = LinearProblem(a, L, bcs=[], 
                                petsc_options={"ksp_type": "preonly", 
                                               "pc_type": "lu"})
        chi_h = problem.solve() 
        # save solution
        a4x.write_mesh(self.filename, mesh)
        a4x.write_function(self.filename, chi_h, name="solution")

        

         

    




