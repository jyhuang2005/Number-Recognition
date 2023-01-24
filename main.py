import math
import sys

import numpy as np
import layer as la
import pixel
import contextlib
import random
import cv2 as cv

with contextlib.redirect_stdout(None):
    import pygame

from pygame.locals import (
    QUIT,
    K_SPACE,
    K_c,
    K_v,
    K_p,
    K_o,
    K_r,
    K_RIGHT,
    K_LEFT,
    K_ESCAPE,
    KEYDOWN
)

sys.setrecursionlimit(30000)

img = cv.imread('medium.png')
assert img is not None, "file could not be read, check with os.path.exists()"
cv.imshow('image window', img)
# add wait key. window waits until user presses a key
cv.waitKey(0)
# and finally destroy/close all open windows
cv.destroyAllWindows()


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

WIDTH, HEIGHT = 560, 560
screen = pygame.display.set_mode((WIDTH, HEIGHT))
screen.fill((255, 255, 255))

drawn_arr = []
processed_arr = []
guess_arr = []
orig_guess_arr = []

# 2d Array of Pixel objects, boundary variables
# twoD_pixelated = [[pixel.Pixel(0)] * 700 for i in range(700)]
twoD_pixels = []
outline_coords = []
borders = []
row = 0
left = WIDTH
right = 0
top = HEIGHT
bottom = 0
recursion = False

x = 0
y = 0
outline_array = [[[-1] for i in range(WIDTH)] for j in range(HEIGHT)]
num_index = 0


def process_image():
    pixels = []
    pxlss = []
    twoD_pixels.clear()
    borders.clear()

    # detecting multiple numbers in one image

    digits = []
    # digit = []
    global left
    global row
    global right
    global top
    global bottom
    checking = True
    vert = 0
    hor = 0

    global outline_array
    global num_index
    outline_array = [[[-1] for _ in range(WIDTH)] for _ in range(HEIGHT)]
    num_index = 0

    for r in range(WIDTH):
        pixels.append([])
        twoD_pixels.append([])
        for c in range(HEIGHT):
            pix = screen.get_at((r, c))[0]
            # pix = img[r, c, 0]
            # print(screen.get_at((r, c))[0], img[r, c, 0])
            pixels[r].append(pix)
            twoD_pixels[r].append(pixel.Pixel(pix))

    # while vert < 700:
    #     for r in range(vert, 700):
    #         if not checking:
    #             checking = True
    #             num_index += 1
    #             break
    #         for c in range(700):
    #             if (r < left - 1 or r > right + 1) or (c < top - 1 or c > bottom + 1):
    #                 if pixels[r][c] == 0:
    #                     row = c
    #                     left = r
    #                     right = 0
    #                     top = 700
    #                     bottom = 0
    #                     global recursion
    #                     checking = False
    #                     recursion = False
    #                     vert = left
    #                     outline_array[c, r] = num_index
    #                     outline_coords.append([c, r])
    #                     find_ccw_neighbor(row, left, 1, 0)
    #                     hor = bottom
    #                     recursion = False
    #                     borders.append([left, right, top, bottom])
    #                     print(left, right, top, bottom)
    #                     break
    #         vert += 1

    global x
    global y
    global recursion
    x = 0
    y = 0

    for r in range(0, WIDTH):
        inside = False
        for c in range(0, HEIGHT):
            if len(outline_array[c][r]) >= 2:
                print(len(outline_array[c][r]))
            switch = outline_array[c][r][0] != -1
            if not inside:
                if switch:
                    inside = True
                    switch = False
            if not inside:
                if pixels[r][c] == 0:
                    inside = True
                    row = c
                    left = r
                    right = 0
                    top = HEIGHT
                    bottom = 0
                    if outline_array[c][r][0] == -1:
                        outline_array[c][r][0] = num_index
                    else:
                        outline_array[c][r].append(num_index)
                        print('hi', outline_array[c][r])
                    outline_coords.append([c, r])
                    recursion = False
                    find_ccw_neighbor(c, r, 1, 0)
                    borders.append([left, right, top, bottom])
                    print(left, right, top, bottom)
                    num_index += 1
            if inside:
                if switch or len(outline_array[c][r]) % 2 == 0:
                    inside = False

    # OG Mean location, min and max
    # min_x = WIDTH
    # max_x = 0
    # min_y = HEIGHT
    # max_y = 0
    # r_sum = 0
    # c_sum = 0
    # pix_count = 0
    # color_sum = 0
    # for r in range(WIDTH):
    #     for c in range(HEIGHT):
    #         pix = pixels[r][c]
    #         # finding mean location of pixels
    #         if pix != 255:
    #             pix_color = (255 - pix) / 255
    #             r_sum += r * pix_color
    #             c_sum += c * pix_color
    #             pix_count += 1
    #             color_sum += pix_color
    #             if min_x is None:
    #                 min_x = r
    #             elif r < min_x:
    #                 min_x = r
    #             if max_x is None:
    #                 max_x = r + 1
    #             elif r + 1 > max_x:
    #                 max_x = r + 1
    #             if min_y is None:
    #                 min_y = c
    #             elif c < min_y:
    #                 min_y = c
    #             if max_y is None:
    #                 max_y = c + 1
    #             elif c + 1 > max_y:
    #                 max_y = c + 1
    # print(min_x, max_x, min_y, max_y)

    print(borders)
    for i in range(0, len(borders)):
        border = borders[i]
        min_y = border[0]
        max_y = border[1] + 1
        min_x = border[2]
        max_x = border[3] + 1
        r_sum = 0
        c_sum = 0
        pix_count = 0
        color_sum = 0

        filtered_pixels = [row[:] for row in pixels]

        for r in range(0, WIDTH):
            inside = False
            for c in range(0, HEIGHT):
                switch = outline_array[c][r][0] == i
                if not inside:
                    if switch:
                        inside = True
                        switch = False
                    else:
                        filtered_pixels[r][c] = 255
                if inside:
                    pix = pixels[r][c]

                    # finding mean location of pixels
                    if pix != 255:
                        pix_color = (255 - pix) / 255
                        r_sum += r * pix_color
                        c_sum += c * pix_color
                        pix_count += 1
                        color_sum += pix_color

                    if switch or len(outline_array[c][r]) % 2 == 0:
                        inside = False

        need_fix = 0
        if pix_count == 0:
            return None, None
        elif color_sum / pow(max(max_x - min_x, max_y - min_y), 2) < 0.15:
            need_fix = 3

        center_x = r_sum // pix_count - WIDTH / 2
        center_y = c_sum // pix_count - HEIGHT / 2

        # print(max(max_x - min_x, max_y - min_y))
        scale = max(max_x - min_x, max_y - min_y) / ((20 - need_fix) * WIDTH / 28)
        scale_constant = WIDTH * (1 - scale) / 2
        # print(scale, scale_constant)

        pixel_size = WIDTH // 28
        pixelated = np.zeros((28, 28))
        total_shade = 0
        for r in range(28):
            for c in range(28):
                av_color = 0
                for a in range(r * pixel_size, (r + 1) * pixel_size):
                    for b in range(c * pixel_size, (c + 1) * pixel_size):
                        true_a = round((a + 0.5) * scale + scale_constant + center_x + 0.5)
                        true_b = round((b + 0.5) * scale + scale_constant + center_y + 0.5)
                        # if r == 6 and c == 6:
                        #     print(center_x + WIDTH // 2, true_a)
                        # if r == 4 or r == 23:
                        #     print(r, c, true_a, true_b)
                        if 0 < true_a < WIDTH and 0 < true_b < HEIGHT:
                            av_color += filtered_pixels[true_a][true_b]
                        else:
                            av_color += 255

                adjusted_shade = (255 - (av_color / (pixel_size ** 2))) / 255
                pixelated[c][r] = adjusted_shade
                # twoD_pixelated[c][r] = pixel.Pixel(adjusted_shade)
                total_shade += adjusted_shade

        # needs fix: after shade adjustment, number too big

        if need_fix == 3:
            for r in range(28):
                for c in range(28):
                    if pixelated[r][c] != 0:
                        pixelated[r][c] = 1

            for r in range(28):
                for c in range(28):
                    if pixelated[r][c] > 0.9:
                        if c > 0 and pixelated[r][c - 1] == 0:
                            pixelated[r][c - 1] = 0.5
                        if c < 27 and pixelated[r][c + 1] == 0:
                            pixelated[r][c + 1] = 0.5
                        if r > 0 and pixelated[r - 1][c] == 0:
                            pixelated[r - 1][c] = 0.5
                        if r < 27 and pixelated[r + 1][c] == 0:
                            pixelated[r + 1][c] = 0.5

        pxls = []
        for pxl in create_one_dimensional(pixelated):
            pxls.append([pxl])
        pxlss.append(np.array(pxls))

    return pxlss, pixels


def is_in_bounds(r, c):
    return -1 < r < len(twoD_pixels) and len(twoD_pixels[0]) > c > -1


# recursively finds the outline of a drawing
def find_ccw_neighbor(r, c, r_dir, c_dir):
    global bottom
    global top
    global left
    global right
    global recursion
    # print(r, c, "O:")
    # print(twoD_pixels[c][r].get_color())
    if r == row and c == left:
        recursion = not recursion
    if recursion:
        if not twoD_pixels[r][c].get_checked():
            twoD_pixels[r][c].set_checked(True)
        if is_in_bounds(r + c_dir, c - r_dir) and twoD_pixels[c - r_dir][r + c_dir].get_color() == 0:
            # print(r + c_dir, c - r_dir, "first")
            # if c - r_dir < left:
            #     left = c - r_dir
            if c - r_dir > right:
                right = c - r_dir
            if r + c_dir < top:
                top = r + c_dir
            if r + c_dir > bottom:
                bottom = r + c_dir

            find_ccw_neighbor(r + c_dir, c - r_dir, c_dir, -r_dir)
        elif is_in_bounds(r + r_dir, c + c_dir) and twoD_pixels[c + c_dir][r + r_dir].get_color() == 0:
            # print(r + r_dir, c + c_dir, "second")
            # if c + c_dir < left:
            #     left = c + c_dir
            if c + c_dir > right:
                right = c + c_dir
            if r + r_dir < top:
                top = r + r_dir
            if r + r_dir > bottom:
                bottom = r + r_dir

            if r_dir == 0:
                if outline_array[r][c][0] == -1:
                    outline_array[r][c][0] = num_index
                else:
                    outline_array[r][c].append(num_index)
                    print('hi', outline_array[r][c])
                outline_coords.append([r, c])
            find_ccw_neighbor(r + r_dir, c + c_dir, r_dir, c_dir)
        elif is_in_bounds(r - c_dir, c + r_dir) and twoD_pixels[c + r_dir][r - c_dir].get_color() == 0:
            # print(r - c_dir, c + r_dir, "third")
            # if c + r_dir < left:
            #     left = c + r_dir
            if c + r_dir > right:
                right = c + r_dir
            if r - c_dir < top:
                top = r - c_dir
            if r - c_dir > bottom:
                bottom = r - c_dir

            if outline_array[r][c][0] == -1:
                outline_array[r][c][0] = num_index
            else:
                outline_array[r][c].append(num_index)
                print('hi', outline_array[r][c])
            outline_coords.append([r, c])
            find_ccw_neighbor(r - c_dir, c + r_dir, -c_dir, r_dir)


def draw_start(r, c):
    pygame.draw.rect(screen, (255, 0, 0), (r, c, 3, 3))


def draw_grid():
    for i in range(28):
        pygame.draw.line(screen, (200, 200, 200), (i * WIDTH // 28, 0), (i * WIDTH // 28, HEIGHT))
    for j in range(28):
        pygame.draw.line(screen, (200, 200, 200), (0, j * HEIGHT // 28), (WIDTH, j * HEIGHT // 28))


def analyze(img):
    l1.update(img)
    l2.update()
    l3.update()

    maxim = 0
    maxim_index = 0
    # print(l3.matrix)
    for j in range(len(l3.matrix)):
        if l3.matrix[j, 0] > maxim:
            maxim = l3.matrix[j, 0]
            maxim_index = j
    return maxim_index


def show_pixelated(num):
    screen.fill((255, 255, 255))
    i = 0

    for r in range(28):
        for c in range(28):

            grayscale_value = max(0, 255 - processed_arr[num][i] * 255)
            if grayscale_value != 255:
                pygame.draw.rect(screen, (grayscale_value, grayscale_value, grayscale_value),
                                 (c * WIDTH // 28 + 1, r * HEIGHT // 28 + 1, WIDTH // 28 - 1, HEIGHT // 28 - 1))
            i += 1


def show_image(num):
    screen.fill((255, 255, 255))

    for r in range(0, WIDTH):
        for c in range(0, HEIGHT):
            if drawn_arr[num][c][r] != 255:
                pygame.draw.rect(screen, (0, 0, 0), (c, r, 1, 1))


def show_outline():
    screen.fill((255, 255, 255))

    for coord in outline_coords:
        pygame.draw.rect(screen, (0, 0, 0), (coord[1], coord[0], 1, 1))


running = True
stroke_size = 25
pygame.display.set_caption(f'Almighty Drawing Canvas - Stroke Size: {stroke_size}')
previous_x, previous_y = 0, 0
first = True
viewing_pix = False
viewing_orig = False
viewing_outline = False
view_num = 0
view_pix_num = 0
num_images_arr = []
count = 0

while running:
    for event in pygame.event.get():
        current_x = pygame.mouse.get_pos()[0]
        current_y = pygame.mouse.get_pos()[1]
        if event.type == KEYDOWN and not pygame.mouse.get_pressed()[0]:
            if event.key == K_SPACE and not (viewing_orig or viewing_pix):
                images, orig_image = process_image()
                num_images_arr.append(len(images))
                if images is not None:
                    drawn_arr.append(orig_image)
                    multi_digit = []
                    for image in images:
                        processed_arr.append(image)
                        analyzed = analyze(image)
                        guess_arr.append(analyzed)
                        multi_digit.append(analyzed)
                    orig_guess_arr.append(multi_digit)
                screen.fill((255, 255, 255))
                first = True
            elif event.key == K_v:
                if not viewing_orig and len(processed_arr) > 0:
                    show_image(view_num)
                    draw_start(left, row)
                    viewing_orig = True
                    viewing_pix = False
                    viewing_outline = False
                elif viewing_orig:
                    viewing_orig = False
                    screen.fill((255, 255, 255))
                    view_num = 0
                    view_pix_num = 0
                    count = 0
            elif event.key == K_o:
                if not viewing_outline and len(processed_arr) > 0:
                    show_outline()
                    draw_start(left, row)
                    viewing_outline = True
                    viewing_orig = False
                    viewing_pix = False
                elif viewing_outline:
                    viewing_outline = False
                    screen.fill((255, 255, 255))
                    view_num = 0
                    view_pix_num = 0
                    count = 0
            elif event.key == K_p:
                if not viewing_pix and len(processed_arr) > 0:
                    show_pixelated(view_pix_num)
                    draw_grid()
                    viewing_pix = True
                    viewing_orig = False
                    viewing_outline = False
                elif viewing_pix:
                    viewing_pix = False
                    screen.fill((255, 255, 255))
                    view_num = 0
                    view_pix_num = 0
                    count = 0
            elif event.key == K_RIGHT and (viewing_orig or viewing_pix or viewing_outline):
                if viewing_pix:
                    if len(processed_arr) > view_pix_num + 1:
                        view_pix_num += 1
                        count += 1
                        if count == len(orig_guess_arr[view_num]):
                            view_num += 1
                            count = 0
                    elif len(processed_arr) > 1:
                        view_pix_num = 0
                        view_num = 0
                        count = 0
                    show_pixelated(view_pix_num)
                    draw_grid()
                else:
                    if len(num_images_arr) > view_num + 1:
                        view_num += 1
                        count = 0
                        v = view_num - 1
                        if v < 0:
                            v = 0
                        for i in range(len(orig_guess_arr[v])):
                            view_pix_num += 1
                    elif len(num_images_arr) > 1:
                        view_num = 0
                        view_pix_num = 0
                        count = 0
                    if viewing_orig:
                        show_image(view_num)
                    elif viewing_outline:
                        show_outline()
            elif event.key == K_LEFT and (viewing_orig or viewing_pix):
                if viewing_pix:
                    if view_pix_num > 0:
                        view_pix_num -= 1
                        count -= 1
                        v = view_num - 1
                        if v < 0:
                            v = 0
                        if count == -1:
                            view_num -= 1
                            count = len(orig_guess_arr[v]) - 1
                    elif len(processed_arr) > 1:
                        view_pix_num = len(processed_arr) - 1
                        view_num = len(num_images_arr) - 1
                        count = len(orig_guess_arr[view_num]) - 1
                    show_pixelated(view_pix_num)
                    draw_grid()
                else:
                    if view_num > 0:
                        view_num -= 1
                        v = view_num - 1
                        if v < 0:
                            v = 0
                        count = len(orig_guess_arr[v]) - 1
                        for i in range(len(orig_guess_arr[view_num])):
                            view_pix_num -= 1
                    elif len(num_images_arr) > 1:
                        view_num = len(num_images_arr) - 1
                        view_pix_num = len(processed_arr) - 1
                        count = len(orig_guess_arr[view_num]) - 1
                    if viewing_orig:
                        show_image(view_num)
                    elif viewing_outline:
                        show_outline()
            elif event.key == K_c and not (viewing_orig or viewing_pix or viewing_outline):
                screen.fill((255, 255, 255))
            elif event.key == K_r and (viewing_orig or viewing_pix or viewing_outline):
                screen.fill((255, 255, 255))
                drawn_arr.clear()
                processed_arr.clear()
                guess_arr.clear()
                orig_guess_arr.clear()
                num_images_arr.clear()
                viewing_orig = False
                viewing_pix = False
                viewing_outline = False
                view_num = 0
                view_pix_num = 0
                count = 0
            elif event.key == K_ESCAPE:
                running = False
            if viewing_orig or viewing_outline:
                guess = ''
                for g in orig_guess_arr[view_num]:
                    guess += str(g)
                pygame.display.set_caption(
                    f'Viewing {view_num + 1}/{len(num_images_arr)} - Guess: {guess}')
            elif viewing_pix:
                pygame.display.set_caption(
                    f'Viewing {view_pix_num + 1}/{len(processed_arr)} - Guess: {guess_arr[view_pix_num]}')
            else:
                pygame.display.set_caption(f'Almighty Drawing Canvas - Stroke Size: {stroke_size}')

        elif pygame.mouse.get_pressed()[0] and not (viewing_orig or viewing_pix or viewing_outline):
            xdis = current_x - previous_x
            ydis = current_y - previous_y
            dis = int(math.sqrt(xdis ** 2 + ydis ** 2))
            pygame.draw.circle(screen, (0, 0, 0), (current_x, current_y), stroke_size)
            if not first:
                for h in range(dis):
                    pygame.draw.circle(screen, (0, 0, 0),
                                       (current_x - ((h * xdis) / dis), current_y - ((h * ydis) / dis)), stroke_size)
            else:
                pygame.draw.circle(screen, (0, 0, 0), (current_x, current_y), stroke_size)
                first = False
        elif event.type == QUIT:
            running = False
        elif event.type == pygame.MOUSEWHEEL and not (viewing_orig or viewing_pix or viewing_outline):
            stroke_size += event.y
            if stroke_size > 500:
                stroke_size = 500
            elif stroke_size < 1:
                stroke_size = 1
            pygame.display.set_caption(f'Almighty Drawing Canvas - Stroke Size: {stroke_size}')
        previous_x, previous_y = current_x, current_y

    pygame.display.flip()

# for img in drawn_arr:
#     print(analyze(img))
