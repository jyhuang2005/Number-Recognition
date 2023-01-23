from copy import deepcopy

import square

board = []
main_queue = []

easy = [[[[0, 0, 0],
          [0, 0, 0],
          [1, 7, 0]],
         [[3, 9, 1],
          [7, 0, 0],
          [4, 0, 0]],
         [[2, 4, 7],
          [1, 9, 0],
          [0, 0, 3]]],
        [[[0, 6, 0],
          [8, 1, 0],
          [4, 0, 0]],
         [[5, 0, 0],
          [0, 7, 0],
          [1, 8, 9]],
         [[0, 0, 1],
          [0, 3, 0],
          [5, 0, 6]]],
        [[[6, 0, 8],
          [0, 0, 0],
          [2, 0, 1]],
         [[9, 0, 7],
          [2, 1, 0],
          [0, 0, 0]],
         [[3, 1, 0],
          [9, 0, 0],
          [6, 0, 0]]]]

medium = [[[[6, 7, 2],
            [0, 0, 0],
            [0, 3, 1]],
           [[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]],
           [[5, 3, 1],
            [0, 0, 0],
            [0, 0, 0]]],
          [[[4, 0, 0],
            [9, 0, 0],
            [0, 6, 3]],
           [[0, 7, 8],
            [5, 0, 0],
            [4, 0, 0]],
           [[0, 0, 2],
            [0, 0, 4],
            [1, 0, 0]]],
          [[[0, 0, 7],
            [0, 0, 6],
            [2, 8, 0]],
           [[0, 0, 0],
            [0, 4, 0],
            [1, 9, 0]],
           [[0, 9, 6],
            [0, 0, 8],
            [0, 5, 0]]]]

hard = [[[[6, 0, 0],
          [0, 0, 0],
          [0, 7, 0]],
         [[0, 5, 0],
          [0, 0, 0],
          [1, 0, 0]],
         [[0, 3, 0],
          [0, 0, 0],
          [0, 9, 0]]],
        [[[2, 0, 0],
          [7, 0, 0],
          [0, 0, 0]],
         [[0, 9, 0],
          [0, 0, 8],
          [0, 0, 0]],
         [[0, 5, 0],
          [0, 0, 9],
          [8, 0, 4]]],
        [[[0, 4, 0],
          [0, 3, 0],
          [0, 1, 0]],
         [[0, 8, 0],
          [0, 6, 0],
          [0, 0, 5]],
         [[0, 0, 0],
          [0, 7, 0],
          [3, 4, 0]]]]

expert = [[[[0, 0, 0],
            [5, 0, 8],
            [0, 0, 0]],
           [[6, 0, 0],
            [0, 0, 0],
            [0, 9, 1]],
           [[0, 0, 0],
            [0, 0, 1],
            [0, 0, 5]]],
          [[[0, 0, 0],
            [3, 4, 0],
            [0, 0, 0]],
           [[0, 0, 0],
            [8, 0, 0],
            [4, 0, 7]],
           [[5, 0, 3],
            [6, 0, 0],
            [0, 0, 0]]],
          [[[9, 0, 0],
            [0, 0, 7],
            [6, 0, 0]],
           [[3, 2, 4],
            [0, 0, 0],
            [0, 0, 0]],
           [[0, 0, 0],
            [0, 1, 0],
            [0, 0, 2]]]]

evil = [[[[9, 0, 0],
          [0, 0, 6],
          [0, 3, 0]],
         [[0, 2, 0],
          [0, 0, 0],
          [9, 0, 1]],
         [[0, 0, 0],
          [0, 8, 0],
          [0, 0, 7]]],
        [[[8, 0, 0],
          [0, 5, 0],
          [0, 0, 0]],
         [[5, 0, 2],
          [0, 4, 0],
          [0, 3, 0]],
         [[0, 7, 0],
          [0, 0, 0],
          [0, 0, 2]]],
        [[[0, 9, 0],
          [3, 0, 0],
          [0, 0, 0]],
         [[7, 0, 5],
          [0, 0, 0],
          [4, 0, 0]],
         [[0, 0, 1],
          [9, 0, 0],
          [0, 0, 0]]]]

evil2 = [[[[9, 0, 2],
           [0, 0, 6],
           [0, 0, 0]],
          [[0, 5, 0],
           [0, 0, 0],
           [0, 0, 4]],
          [[0, 1, 8],
           [0, 0, 0],
           [0, 0, 5]]],
         [[[8, 0, 9],
           [0, 6, 0],
           [0, 4, 0]],
          [[0, 0, 3],
           [0, 9, 0],
           [0, 0, 0]],
          [[2, 0, 0],
           [0, 0, 0],
           [0, 8, 0]]],
         [[[0, 8, 0],
           [1, 0, 5],
           [0, 0, 0]],
          [[0, 0, 0],
           [0, 4, 0],
           [7, 0, 0]],
          [[0, 0, 0],
           [0, 2, 0],
           [3, 0, 0]]]]

evil3 = [[[[0, 1, 0],
           [0, 2, 0],
           [0, 0, 6]],
          [[0, 0, 0],
           [0, 3, 0],
           [4, 0, 0]],
          [[0, 0, 8],
           [0, 0, 0],
           [1, 9, 0]]],
         [[[0, 0, 0],
           [0, 0, 8],
           [0, 6, 0]],
          [[5, 0, 0],
           [0, 0, 0],
           [0, 0, 2]],
          [[0, 0, 7],
           [0, 0, 4],
           [8, 5, 0]]],
         [[[0, 0, 0],
           [0, 0, 1],
           [9, 0, 0]],
          [[0, 0, 0],
           [9, 0, 0],
           [0, 0, 4]],
          [[0, 7, 0],
           [5, 6, 0],
           [0, 0, 0]]]]

uh_oh = [[[[8, 0, 0],
           [0, 0, 3],
           [0, 7, 0]],
          [[0, 0, 0],
           [6, 0, 0],
           [0, 9, 0]],
          [[0, 0, 0],
           [0, 0, 0],
           [2, 0, 0]]],
         [[[0, 5, 0],
           [0, 0, 0],
           [0, 0, 0]],
          [[0, 0, 7],
           [0, 4, 5],
           [1, 0, 0]],
          [[0, 0, 0],
           [7, 0, 0],
           [0, 3, 0]]],
         [[[0, 0, 1],
           [0, 0, 8],
           [0, 9, 0]],
          [[0, 0, 0],
           [5, 0, 0],
           [0, 0, 0]],
          [[0, 6, 8],
           [0, 1, 0],
           [4, 0, 0]]]]

seventeen_clues = [[[[0, 0, 0],
           [0, 6, 0],
           [0, 0, 0]],
          [[1, 0, 0],
           [0, 0, 0],
           [0, 3, 5]],
          [[4, 0, 0],
           [1, 0, 0],
           [0, 0, 0]]],
         [[[0, 1, 0],
           [0, 0, 0],
           [0, 0, 0]],
          [[4, 0, 0],
           [0, 2, 0],
           [0, 0, 0]],
          [[0, 0, 0],
           [0, 7, 0],
           [0, 3, 0]]],
         [[[2, 0, 0],
           [0, 0, 0],
           [3, 0, 7]],
          [[0, 0, 0],
           [8, 0, 0],
           [0, 0, 0]],
          [[0, 5, 0],
           [6, 0, 0],
           [0, 0, 0]]]]

seventeen_clues2 = [[[[5, 0, 0],
           [6, 0, 0],
           [0, 0, 0]],
          [[0, 0, 0],
           [1, 0, 9],
           [0, 0, 0]],
          [[4, 7, 0],
           [0, 0, 0],
           [0, 8, 0]]],
         [[[0, 0, 0],
           [0, 0, 7],
           [0, 8, 0]],
          [[2, 0, 0],
           [0, 0, 0],
           [0, 7, 0]],
          [[0, 0, 6],
           [3, 0, 0],
           [0, 0, 0]]],
         [[[2, 9, 0],
           [0, 0, 0],
           [0, 0, 0]],
          [[0, 0, 0],
           [6, 4, 0],
           [0, 0, 0]],
          [[0, 0, 0],
           [0, 0, 0],
           [0, 0, 0]]]]

# board indices used are either i, j, r, c OR x1, y1, x2, y2 OR a combination of both


def create_board(nums):
    for x1 in range(3):
        board.append([])
        for y1 in range(3):
            board[x1].append([])
            for x2 in range(3):
                board[x1][y1].append([])
                for y2 in range(3):
                    num = nums[x1][y1][x2][y2]
                    if num != 0:
                        board[x1][y1][x2].append(square.Square(num))
                        main_queue.append((x1, y1, x2, y2, num))
                    else:
                        board[x1][y1][x2].append(square.Square())


def elim_pbls(temp_board, queue):
    invalid = 0

    while len(queue) > 0:
        if invalid >= 2:
            return
        invalid = 0
        (x1, y1, x2, y2, val) = queue[0]  # (0-3, 0-3, 0-3, 0-3, 1-9)
        # print((x1, y1, x2, y2, val))

        # elim box
        for r in range(3):
            for c in range(3):
                new_val = temp_board[x1][y1][r][c].elim(val)
                if new_val == -1:
                    invalid += 1
                elif new_val != 0:
                    queue.append((x1, y1, r, c, new_val))

        # elim row
        for j in range(3):
            if j != y1:
                for c in range(3):
                    new_val = temp_board[x1][j][x2][c].elim(val)
                    if new_val == -1:
                        invalid += 2
                    elif new_val != 0:
                        queue.append((x1, j, x2, c, new_val))

        # elim col
        for i in range(3):
            if i != x1:
                for r in range(3):
                    new_val = temp_board[i][y1][r][y2].elim(val)
                    if new_val == -1:
                        invalid += 2
                    elif new_val != 0:
                        queue.append((i, y1, r, y2, new_val))

        queue.pop(0)

    if guess(temp_board):
        return True


def guess(temp_board):
    # find easiest square to guess
    max_val = 10
    max_index = (0, 0, 0, 0)
    for x1 in range(3):
        for y1 in range(3):
            for x2 in range(3):
                for y2 in range(3):
                    if temp_board[x1][y1][x2][y2].solved == 0 and len(temp_board[x1][y1][x2][y2].pbls) <= max_val:
                        max_val = len(temp_board[x1][y1][x2][y2].pbls)
                        max_index = (x1, y1, x2, y2)

    print(max_val)
    # return if found none
    if max_val == 10:
        global board
        print('done')
        board = temp_board
        return True

    (x1, y1, x2, y2) = max_index

    for n in range(max_val):
        board_instance = deepcopy(temp_board)
        board_instance[x1][y1][x2][y2].solve(temp_board[x1][y1][x2][y2].pbls[n])
        temp_queue = [(x1, y1, x2, y2, temp_board[x1][y1][x2][y2].pbls[n])]
        if elim_pbls(board_instance, temp_queue):
            return True


board_nums = seventeen_clues2  # difficulty
create_board(board_nums)
elim_pbls(board, main_queue)

for i in range(3):
    for j in range(3):
        for r in range(3):
            for c in range(3):
                board_nums[i][j][r][c] = board[i][j][r][c].solved

for i in range(3):
    for r in range(3):
        print(board_nums[i][0][r], board_nums[i][1][r], board_nums[i][2][r])
