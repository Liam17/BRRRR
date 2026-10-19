import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from numba import njit


@njit
def mandelbrot(c, max_iter):
    z = 0j
    for n in range(max_iter):
        z=z*z+c

        if abs(z) > 2:
            return n+1-np.log2(np.log2(abs(z)))
    return np.nan


@njit
def julia(z, c, max_iter):
    for n in range(max_iter):
        z=z*z+c

        if abs(z) > 2:
            return n+1-np.log2(np.log2(abs(z)))
    return np.nan


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

    fig,ax = plt.subplots(figsize=(8,8))
    image = np.zeros((height, width))

    bounds = [xmin,xmax,ymin,ymax]
    current_iter=max_iter
    def update():
        xmin,xmax,ymin,ymax=bounds

        for x in range(width):
            for y in range(height):
                re = xmin+(x/width)*(xmax-xmin)
                im = ymin+(y/height)*(ymax-ymin)
                z = complex(re,im)

                if c_param is not None:
                    image[y,x]=fn(z,c_param,current_iter,**kwargs)
                else:
                    image[y,x] = fn(z,current_iter)
        
        img.set_data(image)
        img.set_clim(vmin=0,vmax=current_iter)
        img.set_extent((xmin,xmax,ymin,ymax))
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
        fig.canvas.draw_idle()
    cmap = plt.get_cmap('turbo').copy()
    cmap.set_bad('black')

    img = ax.imshow(
        image,cmap=cmap,
        extent=(xmin,xmax,ymin,ymax),
        origin='lower'
    )

    ax.set_title(fn.__name__)

    def on_key(event):
        nonlocal current_iter
        print(current_iter)
        xmin,xmax,ymin,ymax = bounds

        dx = xmax-xmin
        dy = ymax-ymin

        move = 0.2
        zoom = 0.5

        if event.key=='left':
            bounds[0]-=dx*move
            bounds[1]-=dx*move
        elif event.key == 'right':
            bounds[0] += dx * move
            bounds[1] += dx * move

        elif event.key == 'up':
            bounds[2] += dy * move
            bounds[3] += dy * move

        elif event.key == 'down':
            bounds[2] -= dy * move
            bounds[3] -= dy * move

        elif event.key in ['+','=']:
            cx = (xmin+xmax)/2
            cy = (ymin+ymax)/2

            bounds[:]=[
                cx-dx*zoom/2,
                cx+dx*zoom/2,
                cy-dy*zoom/2,
                cy+dy*zoom/2
            ]

        elif event.key == '-':
            cx=(xmin+xmax)/2
            cy=(ymin+ymax)/2

            bounds[:] = [
                cx-dx/zoom/2,
                cx+dx/zoom/2,
                cy-dy/zoom/2,
                cy+dy/zoom/2
            ]

        elif event.key == 'r':
            current_iter +=100

        elif event.key =='t':
            current_iter-=100

        else:
            return

        update()

    fig.canvas.mpl_connect('key_press_event',on_key)

    update()
    plt.show()

if __name__ == "__main__":
    plot_fractal(mandelbrot, -2.5, 1.5, -2, 2)
    plot_fractal(julia, -2, 2, -2, 2, c_param=complex(-0.8, 0.156))
    # plot_fractal(burningship, -2.2, 1.2, -2.5, 1.5)
    # plot_fractal(multibrot, -2, 2, -2, 2, power=4)
