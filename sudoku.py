import square

board = []
queue = []
board_nums = [[[[0, 0, 0],
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


# board indices used are either i, j, r, c OR x1, y1, x2, y2


def create_board(nums):
    for i in range(3):
        board.append([])
        for j in range(3):
            board[i].append([])
            for r in range(3):
                board[i][j].append([])
                for c in range(3):
                    num = nums[i][j][r][c]
                    if num != 0:
                        board[i][j][r].append(square.Square(num))
                        queue.append((i, j, r, c, num))
                    else:
                        board[i][j][r].append(square.Square())


create_board(board_nums)

while len(queue) > 0:
    (x1, y1, x2, y2, val) = queue[0]  # (0, 1, 2, 3)
    check_square = board[x1][y1][x2][y2]
    # print((x1, y1, x2, y2, val))

    # elim box
    for r in range(3):
        for c in range(3):
            new_val = board[x1][y1][r][c].elim(val)
            if new_val != 0:
                board_nums[x1][y1][r][c] = new_val
                queue.append((x1, y1, r, c, new_val))

    # elim row
    for j in range(3):
        if j != y1:
            for c in range(3):
                new_val = board[x1][j][x2][c].elim(val)
                if new_val != 0:
                    board_nums[x1][j][x2][c] = new_val
                    queue.append((x1, j, x2, c, new_val))

    # elim col
    for i in range(3):
        if i != x1:
            for r in range(3):
                new_val = board[i][y1][r][y2].elim(val)
                if new_val != 0:
                    board_nums[i][y1][r][y2] = new_val
                    queue.append((i, y1, r, y2, new_val))

    queue.pop(0)

# for i in range(3):
#     for j in range(3):
#         for r in range(3):
#             for c in range(3):
#                 print(board[i][j][r][c].solved)

# print(board_nums)

for i in range(3):
    for r in range(3):
        print(board_nums[i][0][r], board_nums[i][1][r], board_nums[i][2][r])
