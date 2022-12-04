
class Pixel:
    def __init__(self, grayscale):
        self.grayscale = grayscale
        self.checked = False

    def get_color(self):
        return self.grayscale

    def get_checked(self):
        return self.checked

    def set_checked(self, checked):
        self.checked = checked
