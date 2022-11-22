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
    K_n,
    K_ESCAPE,
    KEYDOWN,
    MOUSEBUTTONUP,
    MOUSEBUTTONDOWN
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
    for r in range(WIDTH):
        pixels.append([])
        for c in range(HEIGHT):
            pix = screen.get_at((r, c))
            pixels[r].append(pix)
            if pix[0] != 255:
                r_sum += r * (255 - pix[0]) / 255
                c_sum += c * (255 - pix[0]) / 255
                pix_count += 1

    center_x = int(r_sum // pix_count - 350)
    center_y = int(c_sum // pix_count - 350)
    # print(center_x, center_y)

    pixel_size = 25
    pixelated = np.zeros((28, 28))
    for r in range(28):
        for c in range(28):
            av_color = 0
            for a in range(r * pixel_size, r * pixel_size + pixel_size):
                for b in range(c * pixel_size, c * pixel_size + pixel_size):
                    if 0 < a + center_x < 700 and 0 < b + center_y < 700:
                        av_color += pixels[a + center_x][b + center_y][0]
                    else:
                        av_color += 255

            pixelated[c][r] = (255 - (av_color / (pixel_size ** 2))) / 255

    # for r in range(WIDTH):
    #     for c in range(HEIGHT):
    #         if 0 < r + center_x < 700 and 0 < c + center_y < 700:
    #             pixelated[c // 25][r // 25] += pixels[r + center_x][c + center_y][0]
    #         else:
    #             pixelated[c // 25][r // 25] += 255
    # for r in range(28):
    #     for c in range(28):
    #         pixelated[r][c] = (255 - (pixelated[r][c] / (pixel_size ** 2))) / 255

    pxls = []
    for pxl in create_one_dimensional(pixelated):
        pxls.append([pxl])

    return np.array(pxls)


running = True
stroke_size = 25
pygame.display.set_caption(f'Almighty Drawing Canvas - Stroke Size: {stroke_size}')
previous_x, previous_y = 0, 0

while running:
    for event in pygame.event.get():
        current_x = pygame.mouse.get_pos()[0]
        current_y = pygame.mouse.get_pos()[1]
        if pygame.mouse.get_pressed()[0]:
            # current_x = pygame.mouse.get_pos()[0]
            # current_y = pygame.mouse.get_pos()[1]
            xdis = previous_x - current_x
            ydis = previous_y - current_y
            dis = int(math.sqrt(xdis ** 2 + ydis ** 2))
            # pygame.draw.circle(screen, (0, 0, 0), (current_x, current_y + 50), stroke_size)
            print(xdis, current_x, previous_x)
            for c in range(dis + 1):
                if dis != 0:
                    pygame.draw.circle(screen, (0, 0, 0), (current_x - ((c * xdis) / dis), current_y - ((c * ydis) / dis)), stroke_size)
                else:
                    pygame.draw.circle(screen, (0, 0, 0), (current_x, current_y), stroke_size)
        previous_x, previous_y = current_x, current_y
        if event.type == QUIT:
            running = False
        elif event.type == KEYDOWN:
            if event.key == K_n:
                drawn_arr.append(process_image())
                screen.fill((255, 255, 255))
            elif event.key == K_ESCAPE:
                running = False
        elif event.type == pygame.MOUSEWHEEL:
            if event.y > 0:
                stroke_size += event.y
            elif event.y < 0:
                stroke_size += event.y
            if stroke_size > 150:
                stroke_size = 150
            if stroke_size < 1:
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
