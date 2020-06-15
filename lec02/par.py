#!/usr/bin/env -S python3 -u

import taichi as ti
import sys

#ti.init(debug=True)
ti.init()

dt_ex       = 1e-3
dt_im       = 1e-2
maxn        = 256
mass        = 1.0
bottom      = 0.2
gravity     = [0, -9.8]
conn_radius = 0.142
springlen   = 0.1
paused      = ti.var(dt=ti.i32, shape=())

nparticles  = ti.var(dt=ti.i32, shape=())
springk     = ti.var(dt=ti.f32, shape=())
damping     = ti.var(dt=ti.f32, shape=())

pos         = ti.Vector(n=2, dt=ti.f32, shape=maxn)
v           = ti.Vector(n=2, dt=ti.f32, shape=maxn)
restlen     = ti.var(dt=ti.f32, shape=(maxn, maxn))

@ti.kernel
def substep_ex():
    n = nparticles[None]
    for i in range(n):
        # Damping
        v[i] *= ti.exp(-dt_ex * damping[None])
        force = ti.Vector(gravity) * mass
        for j in range(n):
            if restlen[i, j] != 0:
                x_ij = pos[j] - pos[i]
                force += springk[None] * (
                        x_ij.norm() - restlen[i, j]) * x_ij.normalized()
        # print("particle", i, "force:", force, "velocity:", v[i])
        v[i] += dt_ex * force / mass

    # Collide with ground
    for i in range(n):
        if pos[i].y < bottom:
            pos[i].y = bottom
            v[i].y = 0

    # Compute new position
    for i in range(n):
        pos[i] += v[i] * dt_ex

@ti.kernel
def substep_im():
    # TODO: implicit Euler
    pass

@ti.kernel
def create_particle(pos_x: ti.f32, pos_y: ti.f32):
    parid = nparticles[None]
    pos[parid] = [pos_x, pos_y]
    v[parid] = [0, 0]
    nparticles[None] += 1
    for i in range(parid):
        dist = (pos[i] - pos[parid]).norm()
        if dist < conn_radius:
            restlen[i, parid] = springlen
            restlen[parid, i] = springlen

def main():
    gui = ti.GUI("mss", (512, 512), 0xd0dcdf)
    springk[None] = 10000
    damping[None] = 10
    paused[None] = 1
    restlen.fill(0)
    v.fill(0)

    create_particle(0.5, 0.5)
    create_particle(0.4, 0.5)
    create_particle(0.4, 0.4)
    while True:
        for ev in gui.get_events(ti.GUI.PRESS):
            if ev.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                sys.exit()
            elif ev.key == ti.GUI.SPACE:
                paused[None] = 1 - paused[None]
            elif ev.key == ti.GUI.LMB:
                create_particle(ev.pos[0], ev.pos[1])
        if not paused[None]:
            for step in range(10):
                substep_ex()
        X = pos.to_numpy()
        gui.circles(X[:nparticles[None]], color=0x9d4bf0, radius=5)
        gui.line(begin=(0.0, bottom), end=(1.0, bottom), color=0x0)
        for i in range(nparticles[None]):
            for j in range(i+1, nparticles[None]):
                if restlen[i, j] == 0:
                    continue
                gui.line(begin=X[i], end=X[j], color=0x3f3f3f3f, radius=2)
        gui.show()

if (__name__ == "__main__"):
    main()
