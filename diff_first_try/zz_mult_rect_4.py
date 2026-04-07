import pydiffvg
import torch
import skimage
import numpy as np

CARPETA_A_GUARDAR = 'results/mult_rect_4/'
EXPERIMENTO = True
# Use GPU if available
pydiffvg.set_use_gpu(torch.cuda.is_available())

canvas_width, canvas_height = 256 ,256
rect_1 = pydiffvg.Rect(p_min = torch.tensor([40.0, 40.0]),
                     p_max = torch.tensor([80.0, 80.0]))

rect_2 = pydiffvg.Rect(p_min = torch.tensor([160.0, 40.0]),
                     p_max = torch.tensor([220.0, 220.0]))

shapes = [rect_1, rect_2]
rect_group_1 = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
                                 fill_color = torch.tensor([0.3, 0.6, 0.3, 1.0]))

rect_group_2 = pydiffvg.ShapeGroup(shape_ids = torch.tensor([1]),
                                 fill_color = torch.tensor([0.1, 0.2, 0.83, 1.0]))

shape_groups = [rect_group_1, rect_group_2]
scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)

render = pydiffvg.RenderFunction.apply
img = render(256, # width
             256, # height
             2,   # num_samples_x
             2,   # num_samples_y
             0,   # seed
             None, # background_image
             *scene_args)
# The output image is in linear RGB space. Do Gamma correction before saving the image.
pydiffvg.imwrite(img.cpu(), f'{CARPETA_A_GUARDAR}/aa_target.png', gamma=2.2)

target = img.clone()
# Move the rect to produce initial guess
# normalize p_min & p_max for easier learning rate

# --- Cuadrante 1: Superior Izquierda ---
p_min_n_1 = torch.tensor([0.0 / 256.0, 0.0 / 256.0], requires_grad=True)
p_max_n_1 = torch.tensor([128.0 / 256.0, 128.0 / 256.0], requires_grad=True)
color_1 = torch.tensor([0.9, 0.2, 0.2, 1.0], requires_grad=True) # Rojo

# --- Cuadrante 2: Superior Derecha ---
p_min_n_2 = torch.tensor([128.0 / 256.0, 0 / 256.0], requires_grad=True)
p_max_n_2 = torch.tensor([256.0/ 256.0, 128.0 / 256.0], requires_grad=True)
color_2 = torch.tensor([0.2, 0.9, 0.2, 1.0], requires_grad=True) # Verde

# --- Cuadrante 3: Inferior Izquierda ---
p_min_n_3 = torch.tensor([0.0 / 256.0, 128.0 / 256.0], requires_grad=True)
p_max_n_3 = torch.tensor([128.0 / 256.0, 256.0 / 256.0], requires_grad=True)
color_3 = torch.tensor([0.2, 0.2, 0.9, 1.0], requires_grad=True) # Azul

# --- Cuadrante 4: Inferior Derecha ---
p_min_n_4 = torch.tensor([128.0 / 256.0, 128.0 / 256.0], requires_grad=True)
p_max_n_4 = torch.tensor([256.0 / 256.0, 256.0 / 256.0], requires_grad=True)
color_4 = torch.tensor([0.9, 0.9, 0.2, 1.0], requires_grad=True) # Amarillo


# Actualización de propiedades para los 4 rectángulos
rect_1.p_min, rect_1.p_max = p_min_n_1 * 256, p_max_n_1 * 256
rect_group_1.fill_color = color_1

rect_2.p_min, rect_2.p_max = p_min_n_2 * 256, p_max_n_2 * 256
rect_group_2.fill_color = color_2

rect_3 = pydiffvg.Rect(p_min=p_min_n_3 * 256,p_max = p_max_n_3 * 256)
rect_group_3 = pydiffvg.ShapeGroup(shape_ids = torch.tensor([2]),fill_color = color_3)

rect_4 = pydiffvg.Rect(p_min=p_min_n_4 * 256,p_max = p_max_n_4 * 256)
rect_group_4 = pydiffvg.ShapeGroup(shape_ids = torch.tensor([3]),fill_color = color_4)

# Asegúrate de incluirlos todos en la serialización
shapes = [rect_1, rect_2, rect_3, rect_4]
shape_groups = [rect_group_1, rect_group_2, rect_group_3, rect_group_4]


scene_args = pydiffvg.RenderFunction.serialize_scene(\
    canvas_width, canvas_height, shapes, shape_groups)
img = render(256, # width
             256, # height
             2,   # num_samples_x
             2,   # num_samples_y
             1,   # seed
             None, # background_image
             *scene_args)
pydiffvg.imwrite(img.cpu(), f'{CARPETA_A_GUARDAR}/init.png', gamma=2.2)

if EXPERIMENTO:
    # Optimize for radius & center
    optimizer = torch.optim.Adam([
        p_min_n_1, p_max_n_1, color_1,
        p_min_n_2, p_max_n_2, color_2,
        p_min_n_3, p_max_n_3, color_3,
        p_min_n_4, p_max_n_4, color_4
    ], lr=1e-2)


    # Run 100 Adam iterations.
    for t in range(300):
        print('iteration:', t)
        optimizer.zero_grad()
        # Forward pass: render the image.
        rect_1.p_min, rect_1.p_max = p_min_n_1 * 256, p_max_n_1 * 256
        rect_group_1.fill_color = color_1
        
        rect_2.p_min, rect_2.p_max = p_min_n_2 * 256, p_max_n_2 * 256
        rect_group_2.fill_color = color_2
        
        rect_3.p_min, rect_3.p_max = p_min_n_3 * 256, p_max_n_3 * 256
        rect_group_3.fill_color = color_3
        
        rect_4.p_min, rect_4.p_max = p_min_n_4 * 256, p_max_n_4 * 256
        rect_group_4.fill_color = color_4 # Corregido de color_3 a color_4
        
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            canvas_width, canvas_height, shapes, shape_groups)
        img = render(256,   # width
                    256,   # height
                    2,     # num_samples_x
                    2,     # num_samples_y
                    t+1,   # seed
                    None, # background_image
                    *scene_args)
        # Save the intermediate render.sigo
        pydiffvg.imwrite(img.cpu(), f'{CARPETA_A_GUARDAR}/iter_{t}.png', gamma=2.2)
        # Compute the loss function. Here it is L2.
        loss = (img - target).pow(2).sum()
        print('loss:', loss.item())

        # Backpropagate the gradients.
        loss.backward()
        optimizer.step()
        

    # Render the final result.
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)
    img = render(256,   # width
                256,   # height
                2,     # num_samples_x
                2,     # num_samples_y
                102,    # seed
                None, # background_image
                *scene_args)
    # Save the images and differences.
    pydiffvg.imwrite(img.cpu(), f'{CARPETA_A_GUARDAR}/final.png')

    # Convert the intermediate renderings to a video.
    from subprocess import call
    call(["ffmpeg", "-framerate", "24", "-i",
        f"{CARPETA_A_GUARDAR}iter_%d.png", "-vb", "20M",
        f"{CARPETA_A_GUARDAR}/out.mp4"])
