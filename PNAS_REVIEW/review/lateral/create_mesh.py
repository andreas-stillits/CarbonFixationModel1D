"""

Module for creating 3D mesh for lateral diffusion problem.

"""

import numpy as np
from pathlib import Path
import gmsh

kernel = gmsh.model.occ

ASPECT_RATIO = 0.025
FILENAME = Path(__file__).parent / "cylinder_min.msh"
MESH_SIZE = 0.002  # global mesh size, chose [0.002, 0.015, 0.15]
OPEN_GUI = True


def main():
    gmsh.initialize()
    gmsh.model.add("cylinder")
    # cylinder parameters
    bottom_surface = (0, 0, 0)
    axis = (0, 0, 1)
    radius = ASPECT_RATIO
    kernel.addCylinder(*bottom_surface, *axis, radius)
    kernel.synchronize()
    # assign physical groups
    # 3D
    volumes = kernel.getEntities(dim=3)
    assert len(volumes) == 1, "Expected one volume"
    gmsh.model.addPhysicalGroup(3, [tag for dim, tag in volumes], 1, name="airspace")
    # 2D
    surfaces = kernel.getEntities(dim=2)
    for dim, tag in surfaces:
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        if np.isclose(com[2], 1.0):
            # top surface
            gmsh.model.addPhysicalGroup(2, [tag], 2, name="top_surface")
        elif np.isclose(com[2], 0.0):
            # bottom surface
            gmsh.model.addPhysicalGroup(2, [tag], 3, name="bottom_surface")
        else:
            # curved surface
            gmsh.model.addPhysicalGroup(2, [tag], 4, name="curved_surface")
    # mesh options - set mesh size globally
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", MESH_SIZE)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", MESH_SIZE)
    kernel.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write(str(FILENAME))
    # open gui
    if OPEN_GUI:
        gmsh.fltk.run()
    gmsh.finalize()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
