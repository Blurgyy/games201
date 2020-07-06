#!/usr/bin/env -S python3 -u

import taichi as ti

ti.init(debug=True)
#ti.init()

# Properties.
simdim = 2  # Dimension of simulation
side = 32  # Number of nodes on the edge of the triangle
#side = 5  # Number of nodes on the edge of the triangle
n_nodes = (1 + side) * side // 2  # Total number of nodes
n_elems = (side - 1) * (side - 1)  # Total number of triangle elements
E = 1000  # Young's modulus
nu = 0.2  # Poisson's ratio
lame_1 = E / (2 * (1 + nu))
lame_2 = (E * nu) / ((1 + nu) * (1 - 2 * nu))
gravity = [0, -9.8]
# bulk_K = E / (2 * (1 - 2 * v))

nodes = ti.Vector(n=simdim, dt=ti.f32,
                  shape=n_nodes)  # Stores coordinates of nodes
verts = ti.var(dt=ti.int32,
               shape=(n_elems,
                      3))  # Stores indices of the 3 vertices on each element


@ti.func
def elem_cnt(x: int) -> int:
    return 2 * x - 3


@ti.kernel
def build():
    for layer in range(side - 1):
        #print("layer", layer, ':', end=" ")
        nodecnt = side - layer
        elemcnt = elem_cnt(side - layer)
        nodebase = (side + nodecnt + 1) * layer // 2
        elembase = (elem_cnt(side) + elem_cnt(side - layer + 1)) * layer // 2
        #print("{} nodes, index starting from {}, element starting from {}".
        #      format(nodecnt, nodebase, elembase))
        for idx in range(0, elemcnt):
            elemid = elembase + idx
            nodeid = nodebase + idx // 2

            if idx & 1:
                verts[elemid, 0] = nodeid + 1
                verts[elemid, 1] = nodeid + nodecnt
                verts[elemid, 2] = nodeid + nodecnt + 1
            else:
                verts[elemid, 0] = nodeid
                verts[elemid, 1] = nodeid + 1
                verts[elemid, 2] = nodeid + nodecnt
                # print(nodeid, nodeid+1, nodeid+nodecnt)


def main():
    build()


if __name__ == "__main__":
    main()
