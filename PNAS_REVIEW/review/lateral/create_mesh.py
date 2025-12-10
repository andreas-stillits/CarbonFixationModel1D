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
    MAXIMUM_DISTANCE = 0.2

    point_tag = kernel.addPoint(0, 0, 0, 1.0)  # tag = 11
    kernel.synchronize()

    field = gmsh.model.mesh.field
    distance_field = field.add("Distance")
    field.setNumbers(distance_field, "NodesList", [point_tag])
    threshold_field = field.add("Threshold")
    field.setNumber(threshold_field, "InField", distance_field)
    field.setNumber(threshold_field, "LcMin", MINIMUM_RESOLUTION)
    field.setNumber(threshold_field, "LcMax", MAXIMUM_RESOLUTION)
    field.setNumber(threshold_field, "DistMin", MINIMUM_DISTANCE)
    field.setNumber(threshold_field, "DistMax", MAXIMUM_DISTANCE)
    field.setAsBackgroundMesh(threshold_field)
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
