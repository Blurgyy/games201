#!/usr/bin/env -S python3 -u

import taichi as ti
import numpy as np
import sys

ti.init(debug=True)
#ti.init()

w, h = 320, 320

img = ti.Vector(n=3, dt=ti.f32, shape=(w, h))
F = ti.Matrix(n=2, m=2, dt=ti.f32, shape=())
canvas = ti.Vector(n=3, dt=ti.f32, shape=(w * 3, h))


def read_image(fname):
    img.from_numpy(ti.imread(fname)[:, :, :3].astype(np.float32) / 255.0)


@ti.func
def clamp(value: int, lo: int, hi: int) -> int:
    return min(hi, max(value, lo))


@ti.kernel
def deform():
    for i, j in img:
        x_rest = ti.Vector([i / h - 0.5, j / w - 0.5])
        x2_def = ti.Vector([i / h - 0.5, j / w - 0.5])
        x_def = F[None] @ x_rest
        x2_rest = F[None].inverse() @ x2_def
        ni = clamp(int((x_def[0] + 0.5) * w), 0, w - 1)
        nj = clamp(int((x_def[1] + 0.5) * h), 0, h - 1)
        ni2 = clamp(int((x2_rest[0] + 0.5) * w), 0, w - 1)
        nj2 = clamp(int((x2_rest[1] + 0.5) * h), 0, h - 1)
        # Left side of canvas shows original image.
        canvas[i, j] = img[i, j]
        # Middle of canvas shows directly deformed image
        canvas[ni + w, nj] = img[i, j]
        # Right side of canvas shows deformed image, calculated with inv(F)
        canvas[i + 2 * w, j] = img[ni2, nj2]


def main():
    if len(sys.argv) < 2:
        raise ValueError("Too few arguments")
    fname = sys.argv[1]
    read_image(fname)
    gui = ti.GUI("Yet Another Deformation Demo", (w * 3, h))
    F[None] = [[2, 0], [0.5, 1]]
    while True:
        deform()
        gui.set_image(canvas.to_numpy())
        gui.show()


if __name__ == "__main__":
    main()
