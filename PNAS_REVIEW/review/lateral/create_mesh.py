"""

Module for creating 3D mesh for lateral diffusion problem.

"""

import argparse
import numpy as np
import gmsh
from review.utils.constants import ThreeDimExploration
import review.utils.paths as paths


kernel = gmsh.model.occ


def main(argv: list[str] | None = None) -> int:
    constants = ThreeDimExploration()
    parser = argparse.ArgumentParser(
        description="Create 3D mesh for lateral diffusion problem."
    )
    parser.add_argument(
        "version",
        type=str,
        choices=[*constants.allowed],
        help="Version of the temporal exploration constants to use.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Do not open the Gmsh GUI after mesh generation.",
    )
    args = parser.parse_args(argv)
    filename = (
        paths.get_base_path(ensure=True) / "meshes" / f"cylinder_{args.version}.msh"
    )

    gmsh.initialize()
    gmsh.model.add("cylinder")
    # cylinder parameters
    bottom_surface = (0, 0, 0)
    axis = (0, 0, 1)
    radius = constants.get_plug_radius(args.version)
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
    # meshing details
    # mesh size fields
    MINIMUM_RESOLUTION = constants.get_stomatal_radius(args.version) / 4.0
    MAXIMUM_RESOLUTION = constants.get_plug_radius(args.version) / 4.0
    MINIMUM_DISTANCE = constants.get_stomatal_radius(args.version) * 4.0
    MAXIMUM_DISTANCE = 0.5  # halfway to top
    inlet_distance = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(inlet_distance, "FacesList", [3])  # bottom surface
    inlet_threshold = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(inlet_threshold, "IField", inlet_distance)
    gmsh.model.mesh.field.setNumber(inlet_threshold, "LcMin", MINIMUM_RESOLUTION)
    gmsh.model.mesh.field.setNumber(inlet_threshold, "LcMax", MAXIMUM_RESOLUTION)
    gmsh.model.mesh.field.setNumber(inlet_threshold, "DistMin", MINIMUM_DISTANCE)
    gmsh.model.mesh.field.setNumber(inlet_threshold, "DistMax", MAXIMUM_DISTANCE)
    #
    gmsh.model.mesh.field.setAsBackgroundMesh(inlet_threshold)

    kernel.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write(str(filename))
    # open gui
    if not args.no_gui:
        gmsh.fltk.run()
    gmsh.finalize()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
