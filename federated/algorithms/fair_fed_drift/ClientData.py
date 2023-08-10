from typing import List


class ClientData:

    def __init__(self, x: List, y: List, s: List):
        self.x = x
        self.y = y
        self.s = s

    def get_partial_data(self, proportion):
        size = int(len(self.x) * proportion)

        return ClientData(self.x[:size], self.y[:size], self.s[:size])
