class Square:
    def __init__(self, solved=0):
        self.solved = solved
        if solved == 0:
            self.pbls = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        else:
            self.pbls = []

    def elim(self, nums):
        for val in nums:
            self.pbls.remove(val)

        if len(self.pbls) == 1:
            self.solved = self.pbls[0]
            return self.solved
        return 0
