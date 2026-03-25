import jax
import jax.numpy as np
import os
import copy

os.environ["CHALK_JAX"] = "1"
from chalk import *
from colour import Color
from IPython.display import HTML
import jax
import optax
import chalk.transform
import numpy as onp
import glob
import matplotlib.pyplot as plt
from functools import partial
from PIL import Image
import random
import chalk.transform as tx
import imageio.v3 as imageio
import base64

eps = 1e-3



def to_color(c):
    """
    Manual implementation of to_color for DiffRast.
    Maps string names to RGB arrays that JAX can manipulate.
    """
    colors = {
        "blue":  np.array([0.0, 0.0, 1.0]),
        "red":   np.array([1.0, 0.0, 0.0]),
        "white": np.array([1.0, 1.0, 1.0]),
        "black": np.array([0.0, 0.0, 0.0]),
        "grey":  np.array([0.5, 0.5, 0.5]),
    }
    return colors.get(c.lower(), colors["black"])




# import PIL
ff_id = 0


def animate_out(images, steps=36, rate=20, **kwargs):
    global ff_id
    # with imageio.get_writer("/tmp/out.gif", loop=0, fps=rate, **kwargs) as writer:
    writer = []
    for im in images:
        image = imageio.imread(im)
        # image = PIL.Image.open(im)
        writer.append(image)
    # writer[0].save('/tmp/out.gif', save_all=True, append_images=writer,
    #               loop=0, duration=200, transparency=0, disposal=0)
    ff_id += 1

    imageio.imwrite(f"animations/out.{ff_id}.gif", writer, loop=0, fps=rate, **kwargs)

    # imageio.mimsave(
    # base = base64.b64encode(open("/tmp/out.gif", 'br').read()).decode('ascii')
    return HTML(
        f"""
  <div style="text-align:center;"><div style="width:90%; margin:auto;"><img src="animations/out.{ff_id}.gif" id='ff{ff_id}'></div></div>
<script>
new Freezeframe({{
  selector: '#ff{ff_id}',
  overlay: true
}});
</script>



  """
    )




grid = rectangle(4, 4).fill_color("white").line_width(0) + (
    make_path([(-2, 0), (2, 0)]) + make_path([(0, -2), (0, 2)])
).line_color("grey").line_width(1)
hgrid = rectangle(10, 10).fill_color("white").line_width(0).align_tl()
bgrid = rectangle(100, 100).fill_color("white").line_width(0).align_tl()




def animate(fn1, steps=36, rate=20, grid=grid, lw=True, **kwargs):
    images = []

    def fn(t):
        im = fn1(t / steps)
        if isinstance(im, tuple):
            im, extra = im
        else:
            extra = empty()

        return hcat(
            [grid + (im.with_envelope(empty()).line_width(2 if lw else None)), extra],
            0.4,
        ).layout(500)

    if getattr(chalk.transform, 'JAX_MODE', False):
        fn = jax.jit(fn)


    for t in range(1, steps):
        out, h, w = fn(t)
        p = f"/tmp/render.{t:03d}.png"
        chalk.backend.cairo.prims_to_file(out, p, h, w)
        images.append(p)
    return animate_out(images)




def inner(j, i, a):
    """
    Creates a single differentiable square representing a numerical value 'a'.
    """
    # Normalize 'a' to a [0, 1] range for opacity/intensity mapping
    op = np.minimum(1.0, np.abs(a))
    
    return (
        # Define a rounded rectangle (width, height, corner_radius)
        rectangle(0.75, 0.75, 0.1)
        .center_xy()
        .translate(i, j)  # Position based on index in the tensor grid
        .fill_color(
            # Diverging Scale: Blue for positive, Red for negative.
            # (1 - op) * 1 blends the color toward white as the value 'a' approaches 0.
            op * np.where(a > 0, to_color("blue"), to_color("red")) + (1 - op) * 1
        )
        .line_width(0))

@jax.jit
def show_affine(aff):
    """

    # An affine transformation is a geometric mapping that preserves points, straight lines, and planes.
    Visualizes a 3x3 affine transformation matrix (e.g., rotation, translation).
    Essential for debugging how the differentiable rasterizer interprets spatial math.
    """
    # Create grid indices: 0,0,0,1,1,1,2,2,2 // 0,1,2,0,1,2,0,1,2
    # This maps a flattened 9-element vector into a 3x3 visual grid.
    a = aff.reshape(-1)
    out = empty()
    for idx in range(9):
        out = out + inner(idx // 3, idx % 3, a[idx])
    
    # Center and scale the final visualization for the notebook display
    return out.center_xy().scale(0.5)

@jax.jit
def show_color(color):
    """
    Visualizes a color vector (e.g., RGB) as a vertical strip of squares.
    """
    # Maps a 3-element vector to a 3x1 grid (rows 0,1,2 at column 0)
    c = color.reshape(-1)
    out = empty()
    for idx in range(3):
        out = out + inner(idx, 0, c[idx])
    return out.center_xy().scale(0.5)

def show_arc(a):
    """
    A utility to force an Arc segment to render as a full 360-degree circle.
    Used in DiffRast to verify path placement and coordinate transforms.
    """
    # Deepcopy to avoid mutating the original geometric path during the trace
    a = copy.deepcopy(a)
    angles = a.segments.angles
    
    # Update the end angle of the first segment to 360 degrees
    # .at[index].set() is the JAX-compliant way to perform 'in-place' updates
    angles = angles.at[0, 1].set(360)
    
    a.segments.angles = angles
    return a



line = arc_seg(V2(1, 1), 1e-3).stroke()




def translate(t: float):
    "t is a float between 0 and 1"
    affine = tx.Affine.translation(V2(0, t))
    aff_array = np.array(tuple(affine))
    return line.apply_transform(affine), show_affine(aff_array)
animate(translate)