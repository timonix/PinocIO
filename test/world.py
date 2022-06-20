import random


class GH:
    world = [[0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]]

    possible = [[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]]

    def get_world(self):
        return self.world[0:4]

    def next_frame(self):
        self.world[4] = self.world[3]
        self.world[3] = self.world[2]
        self.world[2] = self.world[1]
        self.world[1] = self.world[0]
        self.world[0] = random.choice(self.possible)

    def action(self, i):
        error = (self.world[4][0] - i[0])**2+(self.world[4][1] - i[1])**2+(self.world[4][2] - i[2])**2+(self.world[4][3] - i[3])**2
        return error

    def get_correct(self):
        return self.world[4]

    def get_visual(self):
        pass
