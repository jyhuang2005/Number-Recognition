class Square:
    def __init__(self, solved=0):
        self.solved = solved
        if self.solved == 0:
            self.pbls = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        else:
            self.pbls = []

    def elim(self, num):
        if self.solved == num:
            return -1
        try:
            self.pbls.remove(num)
        except:
            hello = 0

        if len(self.pbls) == 1:
            self.solved = self.pbls[0]
            self.pbls.clear()
            return self.solved
        return 0

    def solve(self, num):
        self.solved = num
        self.pbls.clear()

