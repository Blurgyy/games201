#!/usr/bin/env -S python3 -u

import sys
import numpy
import taichi as ti
ti.init(ti.gpu)

# https://www.shadertoy.com/new

dim = 768
w, h = wh = (dim, dim)
res = ti.Vector([w, h])
pixels = ti.Vector(n=3, dt=ti.f32, shape=wh)

@ti.func
def setPixel(i: ti.i32, j: ti.i32, iTime: ti.f32):
    coord = ti.Vector([i, j])
    uv = coord / res
    uvxyx = ti.Vector([uv[0], uv[1]+2, uv[0]+4])
    color = 0.5 + 0.5 * ti.cos(iTime + uvxyx)
    pixels[i, j] = color

@ti.kernel
def paint(iTime: ti.f32):
    for i, j in pixels:
        setPixel(i, j, iTime)

def main():
    gui = ti.GUI("test", res=wh)
    print("Starting")
    for t in range(int(1e9)):
        if gui.get_event(ti.GUI.ESCAPE):
            sys.exit()
        paint(t/1e2)
        gui.set_image(pixels.to_numpy())
        gui.show()

if __name__ == "__main__":
    main()
