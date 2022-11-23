import math

import idx2numpy
import numpy as np
import layer as la
from sklearn.metrics import mean_squared_error
import contextlib
with contextlib.redirect_stdout(None):
    import pygame

from pygame.locals import (
    QUIT,
    K_SPACE,
    K_ESCAPE,
    KEYDOWN
)


def create_one_dimensional(arr):
    return np.array(arr).ravel()


def get_weights(layer_num):
    if layer_num == 1:
        return np.loadtxt("l1weights.txt")
    elif layer_num == 2:
        return np.loadtxt("l2weights.txt")
    elif layer_num == 3:
        return np.loadtxt("l3weights.txt")


def get_biases(layer_num):
    if layer_num == 1:
        return np.loadtxt("l1biases.txt")
    elif layer_num == 2:
        return np.loadtxt("l2biases.txt")
    elif layer_num == 3:
        return np.loadtxt("l3biases.txt")


l1 = la.Layer(100, weights=get_weights(1), biases=np.rot90([get_biases(1)], 3))
l2 = la.Layer(100, l1, weights=get_weights(2), biases=np.rot90([get_biases(2)], 3))
l3 = la.Layer(10, l2, weights=get_weights(3), biases=np.rot90([get_biases(3)], 3))


WIDTH, HEIGHT = 700, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
screen.fill((255, 255, 255))

drawn_arr = []


def process_image():
    pixels = []

    r_sum = 0
    c_sum = 0
    pix_count = 0
    min_x = None
    max_x = None
    min_y = None
    max_y = None
    for r in range(WIDTH):
        pixels.append([])
        for c in range(HEIGHT):
            pix = screen.get_at((r, c))
            pixels[r].append(pix)
            if pix[0] != 255:
                r_sum += r * (255 - pix[0]) / 255
                c_sum += c * (255 - pix[0]) / 255
                pix_count += 1
                if min_x is None:
                    min_x = r
                elif r < min_x:
                    min_x = r
                if max_x is None:
                    max_x = r
                elif r > max_x:
                    max_x = r
                if min_y is None:
                    min_y = c
                elif c < min_y:
                    min_y = c
                if max_y is None:
                    max_y = c
                elif c > max_y:
                    max_y = c

    if pix_count == 0:
        return None

    center_x = r_sum // pix_count - WIDTH/2
    center_y = c_sum // pix_count - HEIGHT/2

    scale = max(max_x - min_x, max_y - min_y) / (20 * WIDTH / 28)
    scale_constant = WIDTH * (1 - scale) / 2

    pixel_size = 25
    pixelated = np.zeros((28, 28))
    needs_adjustment = True
    total_shade = 0
    pixel_count = 0
    for r in range(28):
        for c in range(28):
            av_color = 0
            for a in range(r * pixel_size, r * pixel_size + pixel_size):
                for b in range(c * pixel_size, c * pixel_size + pixel_size):
                    real_a = int(a * scale + scale_constant + center_x)
                    real_b = int(b * scale + scale_constant + center_y)
                    if 0 < real_a < WIDTH and 0 < real_b < HEIGHT:
                        av_color += pixels[real_a][real_b][0]
                    else:
                        av_color += 255

            adjusted_shade = (255 - (av_color / (pixel_size ** 2))) / 255
            pixelated[c][r] = adjusted_shade
            total_shade += adjusted_shade
            if adjusted_shade == 1:
                needs_adjustment = False

    if needs_adjustment:
        for r in range(28):
            for c in range(28):
                if pixelated[r][c] != 0:
                    pixelated[r][c] = 2

    pxls = []
    for pxl in create_one_dimensional(pixelated):
        pxls.append([pxl])

    return np.array(pxls)


running = True
stroke_size = 25
pygame.display.set_caption(f'Almighty Drawing Canvas - Stroke Size: {stroke_size}')
previous_x, previous_y = 0, 0
first = True

while running:
    for event in pygame.event.get():
        current_x = pygame.mouse.get_pos()[0]
        current_y = pygame.mouse.get_pos()[1]

        if event.type == KEYDOWN and not pygame.mouse.get_pressed()[0]:
            if event.key == K_SPACE:
                image = process_image()
                if image is not None:
                    drawn_arr.append(image)
                screen.fill((255, 255, 255))
                first = True
            elif event.key == K_ESCAPE:
                running = False

        elif pygame.mouse.get_pressed()[0]:
            xdis = current_x - previous_x
            ydis = current_y - previous_y
            dis = int(math.sqrt(xdis ** 2 + ydis ** 2))
            pygame.draw.circle(screen, (0, 0, 0), (current_x, current_y), stroke_size)
            if not first:
                for c in range(dis):
                    pygame.draw.circle(screen, (0, 0, 0), (current_x - ((c * xdis) / dis), current_y - ((c * ydis) / dis)), stroke_size)
            else:
                pygame.draw.circle(screen, (0, 0, 0), (current_x, current_y), stroke_size)
                first = False
        previous_x, previous_y = current_x, current_y

        if event.type == QUIT:
            running = False
        elif event.type == pygame.MOUSEWHEEL:
            stroke_size += event.y
            if stroke_size > 75:
                stroke_size = 75
            elif stroke_size < 1:
                stroke_size = 1

            pygame.display.set_caption(f'Almighty Drawing Canvas - Stroke Size: {stroke_size}')

    pygame.display.flip()


for i in range(len(drawn_arr)):
    l1.update(drawn_arr[i])
    l2.update()
    l3.update()

    maxim = 0
    maxim_index = 0
    for j in range(len(l3.matrix)):
        if l3.matrix[j, 0] > maxim:
            maxim = l3.matrix[j, 0]
            maxim_index = j
    print(maxim_index)
