import math
import sys

import numpy as np
import layer as la
import pixel
import contextlib

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
processed_arr = []
guess_arr = []

# 2d Array of Pixel objects, boundary variables
# twoD_pixelated = [[pixel.Pixel(0)] * 700 for i in range(700)]
twoD_pixelated = []
row = 0
left = 0
right = 0
top = 28
bottom = 0
recursion = False


def process_image():
    sys.setrecursionlimit(10000)
    global twoD_pixelated
    pixels = []

    r_sum = 0
    c_sum = 0
    pix_count = 0
    color_sum = 0
    min_x = None
    max_x = None
    min_y = None
    max_y = None
    for r in range(WIDTH):
        pixels.append([])
        twoD_pixelated.append([])
        for c in range(HEIGHT):
            pix = screen.get_at((r, c))[0]
            pixels[r].append(pix)
            twoD_pixelated[r].append(pixel.Pixel(pix))
            if pix != 255:
                # print(pix)
                pix_color = (255 - pix) / 255
                r_sum += r * pix_color
                c_sum += c * pix_color
                pix_count += 1
                color_sum += pix_color
                if min_x is None:
                    min_x = r
                elif r < min_x:
                    min_x = r
                if max_x is None:
                    max_x = r + 1
                elif r + 1 > max_x:
                    max_x = r + 1
                if min_y is None:
                    min_y = c
                elif c < min_y:
                    min_y = c
                if max_y is None:
                    max_y = c + 1
                elif c + 1 > max_y:
                    max_y = c + 1

    need_fix = 0
    if pix_count == 0:
        return None, None
    elif color_sum / pow(max(max_x - min_x, max_y - min_y), 2) < 0.1:
        need_fix = 2

    center_x = r_sum // pix_count - WIDTH / 2
    center_y = c_sum // pix_count - HEIGHT / 2

    # print(max(max_x - min_x, max_y - min_y))
    scale = max(max_x - min_x, max_y - min_y) / ((20 - need_fix) * WIDTH / 28)
    scale_constant = WIDTH * (1 - scale) / 2
    # print(scale, scale_constant)

    pixel_size = 25
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
                        av_color += pixels[true_a][true_b]
                    else:
                        av_color += 255

            adjusted_shade = (255 - (av_color / (pixel_size ** 2))) / 255
            pixelated[c][r] = adjusted_shade
            # twoD_pixelated[c][r] = pixel.Pixel(adjusted_shade)
            total_shade += adjusted_shade

    # needs fix: after shade adjustment, number too big

    if need_fix == 2:
        for r in range(28):
            for c in range(28):
                if pixelated[r][c] != 0:
                    pixelated[r][c] = 1

        for r in range(28):
            for c in range(28):
                if pixelated[r][c] > 0.9:
                    if c > 0 and pixelated[r][c - 1] == 0:
                        pixelated[r][c - 1] = 0.9
                    if c < 27 and pixelated[r][c + 1] == 0:
                        pixelated[r][c + 1] = 0.9
                    if r > 0 and pixelated[r - 1][c] == 0:
                        pixelated[r - 1][c] = 0.9
                    if r < 27 and pixelated[r + 1][c] == 0:
                        pixelated[r + 1][c] = 0.9

    # print(min_x, max_x, min_y, max_y)


    # attemping to put single digits into an array. Don't need

    digits = []
    # digit = []
    checking = True
    vert = 0
    # while vert <= 700:
    for r in range(vert, 700):
        if not checking:
            break
        for c in range(700):
            if pixels[c][r] < 0.5:
                global left
                global row
                global right
                global top
                global bottom
                global recursion
                left = r
                row = c
                checking = False
                # check_neighbor_similarity(row, left)
                print(left, row, ">:(")
                recursion = False
                find_ccw_neighbor(left, row, 1, 0)
                print(outline_coords)
                print(len(outline_coords))
                # digit = [[pixel.Pixel(0)] * (bottom - top + 1) for i in range(right - left + 1)]
                # for i in range(len(digit)):
                #     for j in range(len(digit[0])):
                #         digit[i][j] = pixelated[top + j][left + i]
                # digits.append(digit)
                vert = right
                # print(digit)
                # print(pixelated[top][left])
                print(left, right, top, bottom)
                # row = 0
                # left = 0
                # right = 0
                # top = 700
                # bottom = 0
                break
        # vert += 1

    pxls = []
    for pxl in create_one_dimensional(pixelated):
        pxls.append([pxl])

    return np.array(pxls), pixels


# might not be necessary
def is_in_bounds(r, c):
    return -1 < r < len(twoD_pixelated) and len(twoD_pixelated[0]) > c > -1


def is_similar(r, c, r2, c2):
    # return abs(twoD_pixelated[r][c].get_color() - twoD_pixelated[r2][c2].get_color() < 0.3)
    return twoD_pixelated[c2][r2].get_color() <= 50


def check_neighbor_similarity(r, c):
    global bottom
    global top
    global left
    global right

    if not twoD_pixelated[r][c].get_checked():
        twoD_pixelated[r][c].set_checked(True)

    # recursion, checking if neighboring pixels are "similar"
    for i in range(-1, 2):
        for j in range(-1, 2):
            if is_in_bounds(r + i, c + j) and is_similar(r, c, r + i, c + j) \
                    and not twoD_pixelated[r + i][c + j].get_checked():
                check_neighbor_similarity(r + i, c + j)
                if c + j > right:
                    right = c + j
                if r + i < top:
                    top = r + i
                if r + i > bottom:
                    bottom = r + i
            print(r + i, c + j)


outline_coords = []

def find_ccw_neighbor(r, c, r_dir, c_dir):
    global bottom
    global top
    global left
    global right
    global recursion
    print(r, c, "O:")
    print(twoD_pixelated[c][r].get_color())
    if r == left and c == row:
        recursion = not recursion
    if recursion:
        if not twoD_pixelated[r][c].get_checked():
            twoD_pixelated[r][c].set_checked(True)
        if is_in_bounds(r + c_dir, c - r_dir) and is_similar(r, c, r + c_dir, c - r_dir):
            print(r + c_dir, c - r_dir, "first")
            find_ccw_neighbor(r + c_dir, c - r_dir, c_dir, -r_dir)
            outline_coords.append([r + c_dir, c - r_dir])
            if c - 1 < left:
                left = c - 1
        elif is_in_bounds(r + r_dir, c + c_dir) and is_similar(r, c, r + r_dir, c + c_dir):
            print(r + r_dir, c + c_dir, "second")
            find_ccw_neighbor(r + r_dir, c + c_dir, r_dir, c_dir)
            outline_coords.append([r + r_dir, c + c_dir])
            if r + 1 > bottom:
                bottom = r + 1
        elif is_in_bounds(r - c_dir, c + r_dir) and is_similar(r, c, r - c_dir, c + r_dir):
            print(r - c_dir, c + r_dir, "third")
            find_ccw_neighbor(r - c_dir, c + r_dir, -c_dir, r_dir)
            outline_coords.append([r - c_dir, c + r_dir])
            if c + 1 > right:
                right = c + 1
        elif is_in_bounds(r - r_dir, c - c_dir) and is_similar(r, c, r - r_dir, c - c_dir):
            print(r - r_dir, c - c_dir, "fourth")
            find_ccw_neighbor(r - r_dir, c - c_dir, -r_dir, -c_dir)
            outline_coords.append([r - r_dir, c - c_dir])
            if r - 1 < top:
                top = r - 1


def draw_outline():
    # screen.fill((255, 255, 255))

    for coord in outline_coords:
        pygame.draw.rect(screen, (0, 0, 0), (coord[1], coord[0], 1, 1))


def draw_start(r, c):
    pygame.draw.rect(screen, (255, 0, 0), (r, c, 3, 3))


def draw_grid():
    for i in range(28):
        pygame.draw.line(screen, (0, 255, 255), (i * 25, 0), (i * 25, 700))
    for j in range(28):
        pygame.draw.line(screen, (0, 255, 255), (0, j * 25), (700, j * 25))


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

    for r in range(0, 700, 25):
        for c in range(0, 700, 25):
            grayscale_value = max(0, 255 - processed_arr[num][i] * 255)
            if grayscale_value != 255:
                pygame.draw.rect(screen, (grayscale_value, grayscale_value, grayscale_value),
                                 (c + 1, r + 1, 25 - 1, 25 - 1))
            i += 1


def show_image(num):
    screen.fill((255, 255, 255))
    for r in range(0, 700):
        for c in range(0, 700):
            if drawn_arr[num][c][r] != 255:
                pygame.draw.rect(screen, (0, 0, 0), (c, r, 1, 1))


running = True
stroke_size = 25
pygame.display.set_caption(f'Almighty Drawing Canvas - Stroke Size: {stroke_size}')
previous_x, previous_y = 0, 0
first = True
viewing_pix = False
viewing_orig = False
view_num = 0


while running:
    for event in pygame.event.get():
        current_x = pygame.mouse.get_pos()[0]
        current_y = pygame.mouse.get_pos()[1]
        if viewing_pix:
            draw_grid()
        if event.type == KEYDOWN and not pygame.mouse.get_pressed()[0]:
            if event.key == K_SPACE and not (viewing_orig or viewing_pix):
                image, orig_image = process_image()
                if image is not None:
                    drawn_arr.append(orig_image)
                    processed_arr.append(image)
                    guess_arr.append(analyze(image))
                screen.fill((255, 255, 255))
                first = True
            elif event.key == K_v:
                if not viewing_orig and len(processed_arr) > 0:
                    show_image(view_num)
                    # draw_outline()
                    draw_start(row, left)
                    viewing_orig = True
                    viewing_pix = False
                elif viewing_orig:
                    viewing_orig = False
                    screen.fill((255, 255, 255))
                    view_num = 0
            elif event.key == K_o:
                if not (viewing_orig or viewing_pix) and len(processed_arr) > 0:
                    draw_outline()
                    draw_start(row, left)
            elif event.key == K_p:
                if not viewing_pix and len(processed_arr) > 0:
                    show_pixelated(view_num)
                    viewing_pix = True
                    viewing_orig = False
                elif viewing_pix:
                    viewing_pix = False
                    screen.fill((255, 255, 255))
                    view_num = 0
            elif event.key == K_RIGHT and (viewing_orig or viewing_pix):
                if len(processed_arr) > view_num + 1:
                    view_num += 1
                    if viewing_orig:
                        show_image(view_num)
                    else:
                        show_pixelated(view_num)
                elif len(processed_arr) > 1:
                    view_num = 0
                    if viewing_orig:
                        show_image(view_num)
                    else:
                        show_pixelated(view_num)
            elif event.key == K_LEFT and (viewing_orig or viewing_pix):
                if view_num > 0:
                    view_num -= 1
                    if viewing_orig:
                        show_image(view_num)
                    else:
                        show_pixelated(view_num)
                elif len(processed_arr) > 1:
                    view_num = len(processed_arr) - 1
                    if viewing_orig:
                        show_image(view_num)
                    else:
                        show_pixelated(view_num)
            elif event.key == K_c and not (viewing_orig or viewing_pix):
                screen.fill((255, 255, 255))
            elif event.key == K_r and (viewing_orig or viewing_pix):
                screen.fill((255, 255, 255))
                drawn_arr.clear()
                processed_arr.clear()
                guess_arr.clear()
                viewing_orig = False
                viewing_pix = False
                view_num = 0
            elif event.key == K_ESCAPE:
                running = False
            if viewing_orig or viewing_pix:
                pygame.display.set_caption(
                    f'Viewing {view_num + 1}/{len(processed_arr)} - Guess: {guess_arr[view_num]}')
            else:
                pygame.display.set_caption(f'Almighty Drawing Canvas - Stroke Size: {stroke_size}')

        elif pygame.mouse.get_pressed()[0] and not (viewing_orig or viewing_pix):
            xdis = current_x - previous_x
            ydis = current_y - previous_y
            dis = int(math.sqrt(xdis ** 2 + ydis ** 2))
            pygame.draw.circle(screen, (0, 0, 0), (current_x, current_y), stroke_size)
            if not first:
                for c in range(dis):
                    pygame.draw.circle(screen, (0, 0, 0),
                                       (current_x - ((c * xdis) / dis), current_y - ((c * ydis) / dis)), stroke_size)
            else:
                pygame.draw.circle(screen, (0, 0, 0), (current_x, current_y), stroke_size)
                first = False
        elif event.type == QUIT:
            running = False
        elif event.type == pygame.MOUSEWHEEL and not (viewing_orig or viewing_pix):
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
