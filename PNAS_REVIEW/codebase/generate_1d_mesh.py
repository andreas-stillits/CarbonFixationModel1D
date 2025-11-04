"""

Generates a 1D mesh for simulations.
Just a simple line segment form z=0 to z=1 with specified number of elements.

"""

import gmsh 

ELEMENTS = 100

def main():
    gmsh.initialize()
    gmsh.model.add("1D_mesh")

    # Define points
    p1 = gmsh.model.geo.addPoint(0, 0, 0)
    p2 = gmsh.model.geo.addPoint(1, 0, 0)
    # Define line
    line = gmsh.model.geo.addLine(p1, p2)
    # Define physical group
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(1, [line], tag=1, name="mesophyll")
    # define boundary point at z=0 as a physical group
    gmsh.model.addPhysicalGroup(0, [p1], tag=2, name="stomatal_interface")
    
    # Set mesh size
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 1.0 / ELEMENTS)

    gmsh.model.mesh.generate(1)
    gmsh.write("./files/1D_mesh.msh")
    
    # open gui
    gmsh.fltk.run()
    
    gmsh.finalize()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())