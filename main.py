import math
import random

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
    K_c,
    K_v,
    K_RIGHT,
    K_LEFT,
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

    scale = max(max_x - min_x, max_y - min_y) / (19.93 * WIDTH / 28)
    scale_constant = WIDTH * (1 - scale) / 2

    pixel_size = 25
    pixelated = np.zeros((28, 28))
    needs_adjustment = 1
    total_shade = 0
    for r in range(28):
        for c in range(28):
            av_color = 0
            for a in range(r * pixel_size + 1, (r + 1) * pixel_size + 1):
                for b in range(c * pixel_size + 1, (c + 1) * pixel_size + 1):
                    true_a = int(a * scale + scale_constant + center_x)
                    true_b = int(b * scale + scale_constant + center_y)
                    # if r == 4 or r == 23:
                    #     print(r, c, true_a, true_b)
                    if 0 < true_a < WIDTH and 0 < true_b < HEIGHT:
                        av_color += pixels[true_a][true_b][0]
                    else:
                        av_color += 255

            adjusted_shade = (255 - (av_color / (pixel_size ** 2))) / 255
            pixelated[c][r] = adjusted_shade
            total_shade += adjusted_shade
            needs_adjustment *= min(1, 1.8 - adjusted_shade)

    if needs_adjustment > 0.001:
        for r in range(28):
            for c in range(28):
                if pixelated[r][c] != 0:
                    pixelated[r][c] = min(1, pixelated[r][c] + needs_adjustment)

        for r in range(28):
            for c in range(28):
                if pixelated[r][c] > 0.3:
                    if c > 0 and pixelated[r][c - 1] == 0:
                        pixelated[r][c - 1] = min(0.3, pixelated[r][c] + needs_adjustment / 2)
                    if c < 27 and pixelated[r][c + 1] == 0:
                        pixelated[r][c + 1] = min(0.3, pixelated[r][c] + needs_adjustment / 2)
                    if r > 0 and pixelated[r - 1][c] == 0:
                        pixelated[r - 1][c] = min(0.3, pixelated[r][c] + needs_adjustment / 2)
                    if r < 27 and pixelated[r + 1][c] == 0:
                        pixelated[r + 1][c] = min(0.3, pixelated[r][c] + needs_adjustment / 2)

    # print(min_x, max_x, min_y, max_y)

    pxls = []
    for pxl in create_one_dimensional(pixelated):
        pxls.append([pxl])

    return np.array(pxls)


def show_pixelated(num):
    screen.fill((255, 255, 255))
    i = 0

    for r in range(0, 700, 25):
        for c in range(0, 700, 25):
            grayscale_value = max(0, 255 - drawn_arr[num][i] * 255)
            for k in range(r + 1, r + 24):
                for l in range(c + 1, c + 24):
                    pygame.draw.rect(screen, (grayscale_value, grayscale_value, grayscale_value), (l, k, 1, 1))
            i += 1


running = True
stroke_size = 25
pygame.display.set_caption(f'Almighty Drawing Canvas - Stroke Size: {stroke_size}')
previous_x, previous_y = 0, 0
first = True
viewing = False
view_num = 0

while running:
    for event in pygame.event.get():
        current_x = pygame.mouse.get_pos()[0]
        current_y = pygame.mouse.get_pos()[1]

        if event.type == KEYDOWN and not pygame.mouse.get_pressed()[0]:
            if event.key == K_SPACE and not viewing:
                image = process_image()
                if image is not None:
                    drawn_arr.append(image)
                screen.fill((255, 255, 255))
                first = True
            elif event.key == K_v:
                if not viewing and len(drawn_arr) > 0:
                    show_pixelated(view_num)
                    viewing = True
                elif viewing:
                    viewing = False
                    screen.fill((255, 255, 255))
                    view_num = 0
            elif event.key == K_RIGHT and viewing:
                if len(drawn_arr) > view_num + 1:
                    view_num += 1
                    show_pixelated(view_num)
            elif event.key == K_LEFT and viewing:
                if view_num > 0:
                    view_num -= 1
                    show_pixelated(view_num)
            elif event.key == K_c and not viewing:
                screen.fill((255, 255, 255))
            elif event.key == K_ESCAPE:
                running = False

        elif pygame.mouse.get_pressed()[0] and not viewing:
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
            if stroke_size > 1000:
                stroke_size = 1000
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
