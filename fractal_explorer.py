import numpy as np
import matplotlib.pyplot as plt
from numba import jit


@jit
def mandelbrot(c, max_iter):
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter


@jit
def julia(z, c, max_iter):
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter


@jit
def burningship(c, max_iter):
    z = 0
    for n in range(max_iter):
        z = complex(abs(z.real), abs(z.imag))
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter


@jit
def multibrot(c, max_iter, power=3):
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z**power + c
    return max_iter


def plot_fractal(fn, xmin, xmax, ymin, ymax, width=800, height=800, max_iter=100, c_param=None, **kwargs):
    image = np.zeros((height, width))
    for x in range(width):
        for y in range(height):
            re = xmin + (x / width) * (xmax - xmin)
            im = ymin + (y / height) * (ymax - ymin)
            z = complex(re, im)
            if c_param is not None:
                image[y, x] = fn(z, c_param, max_iter, **kwargs)
            else:
                image[y, x] = fn(z if fn == julia else complex(re, im), max_iter)
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='hot', extent=(xmin, xmax, ymin, ymax))
    plt.title(fn.__name__)
    plt.show()


if __name__ == "__main__":
    plot_fractal(mandelbrot, -2.5, 1.5, -2, 2)
    plot_fractal(julia, -2, 2, -2, 2, c_param=complex(-0.8, 0.156))
    plot_fractal(burningship, -2.2, 1.2, -2.5, 1.5)
    plot_fractal(multibrot, -2, 2, -2, 2, power=4)
