#!/usr/bin/env -S python3 -u
import taichi as ti
import random

ti.init()

n = 30
it = 50

A = ti.var(dt=ti.f32, shape=(n, n))
x = ti.var(dt=ti.f32, shape=n)
nx = ti.var(dt=ti.f32, shape=n)
b = ti.var(dt=ti.f32, shape=n)


@ti.kernel
def iterate():
    for i in range(n):
        rhs = b[i]
        for j in range(n):
            if (j != i):
                rhs -= A[i, j] * x[j]
        nx[i] = rhs / A[i, i]
    for i in range(n):
        x[i] = nx[i]


@ti.kernel
def residual() -> ti.f32:
    res = 0.0
    for i in range(n):
        rhs = b[i] * 1.0
        for j in range(n):
            rhs -= A[i, j] * x[j]
        res += rhs * rhs
    return res


def main():
    # Initialize
    for i in range(n):
        for j in range(n):
            A[i, j] = random.random() - 0.5
        A[i, i] += n * 0.1
        b[i] = random.random() * 100
    # Iterate $it times
    for i in range(it):
        iterate()
        print(f'iter {i}, residual = {residual():0.10f}')
    for i in range(n):
        lhs = 0.0
        for j in range(n):
            lhs += A[i, j] * x[j]
        assert (abs(lhs - b[i]) < 1e-4)


if (__name__ == '__main__'):
    main()
