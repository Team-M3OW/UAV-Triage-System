#!/usr/bin/python3
# Solves a raster using classic search algorithms on a pre-computed cost grid.

import numpy as np
import heapq

# --- This class remains unchanged ---
class PriorityQueue:
    def __init__(self):
        self.elements = []
    def empty(self):
        return len(self.elements) == 0
    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))
    def get(self):
        return heapq.heappop(self.elements)[1]

# --- This function remains unchanged ---
def getNeighbors(i, m, n, cost_grid, ntype = 4):
    neighbors = []
    rows, cols = cost_grid.shape
    potential_neighbors = []
    if ntype >= 4:
        potential_neighbors.extend([(i[0] - 1, i[1]), (i[0] + 1, i[1]), (i[0], i[1] - 1), (i[0], i[1] + 1)])
    if ntype >= 8:
        potential_neighbors.extend([(i[0] - 1, i[1] - 1), (i[0] - 1, i[1] + 1), (i[0] + 1, i[1] - 1), (i[0] + 1, i[1] + 1)])
    # (16-way neighborhood code can be added here if needed)

    for r, c in potential_neighbors:
        if 0 <= r < m and 0 <= c < n and cost_grid[r, c] < np.inf:
            neighbors.append((r, c))
    return neighbors


def solve(grid, start, solver=0, ntype=4, trace=False):
    # --- MODIFICATION: The 'goal' parameter has been removed from the function signature.
    rows, cols = grid.shape
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {start: None}
    cost_so_far = {start: 0}

    tgrid = None
    if trace:
        tgrid = np.zeros_like(grid)

    # Explore the entire reachable grid
    while not frontier.empty():
        current = frontier.get()

        if trace:
            tgrid[current] = 0.25

        # --- MODIFICATION: The 'if current == goal: break' statement has been REMOVED. ---
        # The search now continues until the entire map is explored.

        for next in getNeighbors(current, rows, cols, grid, ntype):
            dist = np.sqrt((next[0] - current[0])**2 + (next[1] - current[1])**2)
            avg_cost_val = (grid[current] + grid[next]) / 2.0
            edge_cost = avg_cost_val * dist
            new_cost = cost_so_far[current] + edge_cost

            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost
                # Note: A* is not used here because there's no single goal for the heuristic.
                # The solver will perform a pure Dijkstra search.
                frontier.put(next, priority)
                came_from[next] = current

    # --- MODIFICATION: The function now returns the complete search maps. ---
    # Path reconstruction will be handled by the calling script.
    return came_from, cost_so_far, tgrid
