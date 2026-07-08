import numpy as np
import heapq


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]


def get_neighbors(i, m, n, cost_grid, ntype=4):
    neighbors = []
    rows, cols = cost_grid.shape
    potential_neighbors = []
    if ntype >= 4:
        potential_neighbors.extend([
            (i[0] - 1, i[1]), (i[0] + 1, i[1]),
            (i[0], i[1] - 1), (i[0], i[1] + 1),
        ])
    if ntype >= 8:
        potential_neighbors.extend([
            (i[0] - 1, i[1] - 1), (i[0] - 1, i[1] + 1),
            (i[0] + 1, i[1] - 1), (i[0] + 1, i[1] + 1),
        ])
    for r, c in potential_neighbors:
        if 0 <= r < m and 0 <= c < n and cost_grid[r, c] < np.inf:
            neighbors.append((r, c))
    return neighbors


def solve(grid, start, ntype=4, trace=False):
    rows, cols = grid.shape
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {start: None}
    cost_so_far = {start: 0}

    tgrid = None
    if trace:
        tgrid = np.zeros_like(grid)

    while not frontier.empty():
        current = frontier.get()

        if trace:
            tgrid[current] = 0.25

        for nxt in get_neighbors(current, rows, cols, grid, ntype):
            dist = np.sqrt((nxt[0] - current[0]) ** 2 + (nxt[1] - current[1]) ** 2)
            avg_cost_val = (grid[current] + grid[nxt]) / 2.0
            edge_cost = avg_cost_val * dist
            new_cost = cost_so_far[current] + edge_cost

            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                frontier.put(nxt, new_cost)
                came_from[nxt] = current

    return came_from, cost_so_far, tgrid
