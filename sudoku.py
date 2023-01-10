import square

board = []
queue = []


def create_board():
    for i in range(3):
        board.append([])
        for j in range(3):
            board[i].append([])
            for r in range(3):
                board[i][j].append([])
                for c in range(3):
                    board[i][j][r].append(square.Square())


while len(queue) != 1:
    (x1, y1, x2, y2) = queue[0] #(0, 1, 2, 3)
    check_square = board[x1][y1][x2][y2]
    check_val = check_square.solved

    #elim box
    for row in board[x1][y1]:
        for square in row:
            square.elim(check_val)

    #elim row


    #elim col



