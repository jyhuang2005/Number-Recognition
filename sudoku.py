import square

board = []
queue = []
for i in range(3):
    board.append([])
    for j in range(3):
        board[i].append([])
        for r in range(3):
            board[i][j].append([])
            for c in range(3):
                board[i][j][r].append(square.Square())


print(board)

while len(queue) != 1:
    (x1, y1, x2, y2) = queue[0] #(0, 1, 2, 3)
    check_square = board[x1][y1][x2][y2]
    check_val = check_square.solved

    #elim box
    for row in range(3):
        for col in range(3):
            if check_square[x1][y2][row][col].elim(check_val) != 0:
                queue.append((x1, y1, row, col))

    #elim row
    for big_col in range(3):
        if big_col != y1:
            for col in range(3):
                if board[x1][big_col][x2][col].elim(check_val) != 0:
                    queue.append((x1, big_col, x2, col))

    #elim col
    for big_row in range(3):
        if big_row != x1:
            for row in range(3):
                if board[big_row][y1][row][y2].elim(check_val) != 0:
                    queue.append((big_row, y1, row, y2))

    queue.pop(0)

print(board)
