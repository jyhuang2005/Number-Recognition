import square

board = []
for i in range(3):
    board.append([])
    for j in range(3):
        board[i].append([])
        for r in range(3):
            board[i][j].append([])
            for c in range(3):
                board[i][j][r].append(square.Square())


print(board)
