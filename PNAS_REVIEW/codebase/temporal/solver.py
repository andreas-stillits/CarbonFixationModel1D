"""

Module to explore the effect of time-dependence on quantities such as gs, Ca and K

"""


from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import pyvista 

import ufl
from dolfinx import mesh, fem, plot
from dolfinx.fem.petsc import create_matrix, create_vector, assemble_vector, assemble_matrix
from typing import Callable
import functools


def get_homogeneous_solution(x: np.ndarray, params: list[float]) -> np.ndarray:
    tau, gamma, chi_ = params 
    return chi_ + (1-chi_)/(1 + (tau/gamma)*np.tanh(tau)) * (np.cosh(tau*(x - 1)))/(np.cosh(tau))


class TemporalSolver:
    def __init__(self, 
                 params: list[float], # [tau, gamma, chi_]
                 filename: str = "temporal_solution.txt",
                 timing: tuple[float, float, float] = (0.0, 1.0, 0.01),
                 domain: mesh.Mesh | None = None, 
                 domain_resolution: int = 100, 
                 functionspace: fem.FunctionSpace | None = None,
                 animate: bool = True,
                 animation_name: str = "chi_time.gif",
                 order: int = 1,
                 update_delta: Callable[[np.ndarray, float], np.ndarray] | None = None,
                 update_kappa: Callable[[np.ndarray, float], np.ndarray] | None = None,
                 update_stomata: Callable[[np.ndarray, float], np.ndarray] | None = None,
                 update_atmospheric: Callable[[np.ndarray, float], np.ndarray] | None = None,
                 initial_condition: Callable | None = None):
        self.tau = params[0]
        self.gamma = params[1]
        self.chi_ = params[2]
        self.filename = filename
        self.t_start = timing[0]
        self.t_end = timing[1]
        self.dt = timing[2]
        self.domain = domain
        self.domain_resolution = domain_resolution
        self.functionspace = functionspace
        self.animate = animate
        self.animation_name = animation_name
        self.update_delta = (lambda x, t: 1.0) if update_delta is None else update_delta
        self.update_kappa = (lambda x, t: 1.0) if update_kappa is None else update_kappa
        self.update_stomata = (lambda x, t: 1.0) if update_stomata is None else update_stomata
        self.update_atmospheric = (lambda x, t: 1.0) if update_atmospheric is None else update_atmospheric
        self.initial_condition = functools.partial(get_homogeneous_solution, params=params) if initial_condition is None else initial_condition
        self.order = order


    def setup_domain(self) -> tuple[ufl.Measure, ufl.Measure]:
        if self.domain is None:
            self.domain = mesh.create_interval(MPI.COMM_WORLD, self.domain_resolution, [0.0, 1.0])
        if self.functionspace is None:
            self.functionspace = fem.functionspace(self.domain, ("CG", self.order))
        # --- Create facet tags --- 
        tdim = self.domain.topology.dim 
        fdim = tdim - 1
        facets_left = mesh.locate_entities(self.domain, fdim, lambda x: np.isclose(x[0], 0.0))
        facets_right = mesh.locate_entities(self.domain, fdim, lambda x: np.isclose(x[0], 1.0))
        facet_indices = np.concatenate([facets_left, facets_right])
        facet_tags = np.concatenate([np.full_like(facets_left, 1), np.full_like(facets_right, 2)])
        mt = mesh.meshtags(self.domain, fdim, facet_indices, facet_tags)
        dx = ufl.Measure("dx", domain=self.domain)
        ds = ufl.Measure("ds", domain=self.domain, subdomain_data=mt)
        return dx, ds 
    

    def setup_plotting(self, chi_n: fem.Function) -> tuple[pyvista.pyvista_ndarray, pyvista.UnstructuredGrid, pyvista.Plotter]:
        grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(self.functionspace))
        xcoords = grid.points[:, 0].copy()
        grid.point_data["chi"] = chi_n.x.array.copy()  
        grid.points = np.c_[xcoords, 
                            grid.point_data["chi"], 
                            np.zeros_like(xcoords)]  # x, chi, 0 

        plotter = pyvista.Plotter(window_size=[500, 400])
        plotter.open_gif(self.animation_name, fps=10)
        actor = plotter.add_mesh(
            grid,
            show_edges=True,
            lighting=False,
            show_scalar_bar=False,
            clim=(0.0, 1.0),
            line_width=4
        )

        homogeneous = grid.copy(deep=True)
        homogeneous.points = np.c_[xcoords, 
                                   get_homogeneous_solution(xcoords, [self.tau, self.gamma, self.chi_]), 
                                   np.zeros_like(xcoords)]
        homogeneous_actor = plotter.add_mesh(
            homogeneous,
            color="black",
            show_edges=False,
            lighting=False,
            line_width=2
        )

        vmin, vmax = 0.0, 1.0 
        bounds = (vmin, vmax, vmin, vmax, 0.0, 0.0)
        cube = plotter.show_grid(
            bounds=bounds,
            location="outer",
            xtitle=r"$\lambda$ (normalized depth)",
            ytitle=r"$\chi$ (normalized concentration)",
            ticks="both",
            font_size=2,
            grid="both"
        )
        cube.SetLabelOffset(4)
        
        plotter.view_xy()
        plotter.set_focus((0.5, 0.5, 0.0))
        plotter.set_position((0.5, 0.5, 10.0))
        plotter.camera.parallel_projection = True
        plotter.camera.SetParallelScale(0.5)
        plotter.camera.zoom(0.8)
        plotter.write_frame()

        return xcoords, grid, plotter

    
    def solve(self):
        # --- get measures and set up domain mesh and functionspace if not supplied ---
        dx, ds = self.setup_domain()
        
        # --- simulation parameters ---
        tau2  = fem.Constant(self.domain, PETSc.ScalarType(self.tau**2))
        gamma = fem.Constant(self.domain, PETSc.ScalarType(self.gamma))
        chi_  = fem.Constant(self.domain, PETSc.ScalarType(self.chi_))
        
        # --- Trial and Test Functions ---
        chi = ufl.TrialFunction(self.functionspace)
        v   = ufl.TestFunction(self.functionspace)
        
        # --- time stepping parameters --- 
        dt        = fem.Constant(self.domain, PETSc.ScalarType(self.dt))
        num_steps = int(np.ceil((self.t_end - self.t_start) / self.dt))
        t         = self.t_start

        # --- initial condition ---
        chi_n            = fem.Function(self.functionspace, name="chi_n")
        x                = self.functionspace.tabulate_dof_coordinates()[:, 0]
        chi_n.x.array[:] = self.initial_condition(x)
        chi_n.x.scatter_forward()

        # --- time dependent coefficients ---
        delta_t       = fem.Function(self.functionspace, name="delta_t")
        kappa_t       = fem.Function(self.functionspace, name="kappa_t")
        stomata_t     = fem.Function(self.functionspace, name="stomata_t")
        atmospheric_t = fem.Function(self.functionspace, name="atmospheric_t")

        # --- variational forms for backwards euler ---
        d, k, s, ca = delta_t, kappa_t, stomata_t, atmospheric_t # aliases for readability in forms
        
        a = chi * v * dx \
            +dt * tau2 * k * chi * v * dx \
            +dt * d * ufl.inner(ufl.grad(chi), ufl.grad(v)) * dx \
            +dt * gamma * s * chi * v * ds(1) # left boundary at lambda = 0
        
        L = chi_n * v * dx \
            +dt * tau2 * k * chi_ * v  * dx \
            +dt * gamma * s * ca * v * ds(1) # left boundary at lambda = 0

        a = fem.form(a)
        L = fem.form(L) 

        # precreate PETSc objects with correct sparsity pattern 
        A = create_matrix(a)
        b = create_vector(L)    
        ksp = PETSc.KSP().create(MPI.COMM_WORLD)
        ksp.setType("cg")          # SPD for this model (with Dirichlet or pure Neumann + mass term)
        ksp.getPC().setType("hypre")
        ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=500)
        chi_new = fem.Function(self.functionspace, name="chi_new")

        # --- create animation objects if requested ---
        if self.animate:
            xcoords, grid, plotter = self.setup_plotting(chi_n)

        # --- Extract time series of rescaled assimilation rate ---        
        times = [] 
        alphas = []
        # dofs_edge = fem.locate_dofs_geometrical(self.functionspace, lambda x: np.isclose(x[0], 0.0))
        # index_edge = int(dofs_edge[0])

        # START INTEGRATION LOOP

        for n in range(num_steps):
            t += self.dt

            # update time-dependent coefficients
            delta_t.x.array[:] = self.update_delta(x, t); delta_t.x.scatter_forward()
            kappa_t.x.array[:] = self.update_kappa(x, t); kappa_t.x.scatter_forward()
            stomata_t.x.array[:] = self.update_stomata(x, t); stomata_t.x.scatter_forward()
            atmospheric_t.x.array[:] = self.update_atmospheric(x, t); atmospheric_t.x.scatter_forward()

            # reassemble A and b due to time-dependence 
            A.zeroEntries() 
            assemble_matrix(A, a)
            A.assemble()

            with b.localForm() as b_local:
                b_local.set(0.0)
            assemble_vector(b, L)

            # Solve A chi_n = b 
            ksp.setOperators(A)
            ksp.solve(b, chi_new.x.petsc_vec)
            
            # update for next time step
            chi_new.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            chi_n.x.array[:] = chi_new.x.array
            chi_n.x.scatter_forward() 

            # --- update plotting --- 
            if self.animate:
                grid.point_data["chi"][:] = chi_new.x.array
                grid.points[:,:] = np.c_[xcoords, 
                                         chi_new.x.array,
                                         np.zeros_like(xcoords)]  # x, chi, 0
                plotter.write_frame()
            
            # --- extract rescaled assimilation rate at this time step ---
            times.append(t)
            # chi_edge = chi_new.x.array[index_edge]
            # stomata_edge = stomata_t.x.array[index_edge] 
            # alpha = self.gamma * stomata_edge * (1 - chi_edge)
            alpha = fem.assemble_scalar(fem.form(kappa_t * (chi_new - chi_)*dx))
            alphas.append(alpha)
            # END INTEGRATION LOOP
        
        for item in [A, b, ksp]:
            item.destroy()
        if self.animate:
            plotter.close()

        data = np.vstack([times, alphas]).T
        np.savetxt(self.filename, data, delimiter=";")

        