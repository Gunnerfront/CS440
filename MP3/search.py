from collections import deque
from functools import lru_cache
import heapq

# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP3. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)


# Feel free to use the code below as you wish
# Initialize it with a list/tuple of objectives
# Call compute_mst_weight to get the weight of the MST with those objectives
# TODO: hint, you probably want to cache the MST value for sets of objectives you've already computed...
# Note that if you want to test one of your search methods, please make sure to return a blank list
#  for the other search methods otherwise the grader will not crash.
class MST:
    #@lru_cache(maxsize=32)
    def __init__(self, objectives):
        self.elements = {key: None for key in objectives}

        # TODO: implement some distance between two objectives
        # ... either compute the shortest path between them, or just use the manhattan distance between the objectives
        self.distances   = {
                (i, j): manhattan_distance(i, j)
                for i, j in self.cross(objectives)
            }

    # Prim's algorithm adds edges to the MST in sorted order as long as they don't create a cycle
    @lru_cache(maxsize=32)
    def compute_mst_weight(self):
        weight      = 0
        for distance, i, j in sorted((self.distances[(i, j)], i, j) for (i, j) in self.distances):
            if self.unify(i, j):
                weight += distance
        return weight

    # helper checks the root of a node, in the process flatten the path to the root
    def resolve(self, key):
        path = []
        root = key
        while self.elements[root] is not None:
            path.append(root)
            root = self.elements[root]
        for key in path:
            self.elements[key] = root
        return root

    # helper checks if the two elements have the same root they are part of the same tree
    # otherwise set the root of one to the other, connecting the trees
    def unify(self, a, b):
        ra = self.resolve(a)
        rb = self.resolve(b)
        if ra == rb:
            return False
        else:
            self.elements[rb] = ra
            return True

    # helper that gets all pairs i,j for a list of keys
    def cross(self, keys):
        return (x for y in (((i, j) for j in keys if i < j) for i in keys) for x in y)

class CellNode:
    def __init__(self, row, col, prev_row, prev_col, dist = 0):
        self.row = row
        self.col = col
        self.prev_row = prev_row
        self.prev_col = prev_col
        self.dist = dist
        self.was_explored = False
    
    def to_tuple(self):
        return tuple((self.row, self.col))
    
    def set_f_n(self, f_n):
        self.f_n = f_n

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    closest_path = []

    first_cell = CellNode(maze.start[0], maze.start[1], -1, -1)
    explored = {first_cell.to_tuple() : first_cell}
    frontier = deque((first_cell.to_tuple(), ))
    
    # go thru all paths
    while frontier:
        curr_cell_as_tuple = tuple(frontier.popleft())
        curr_cell_maze = maze[curr_cell_as_tuple]
        if curr_cell_maze == maze.legend.waypoint:
            break
        else:
            # add neighbors if not at waypoint
            for neighbor in maze.neighbors(*curr_cell_as_tuple):
                neighbor_as_cell = CellNode(neighbor[0], neighbor[1], curr_cell_as_tuple[0], curr_cell_as_tuple[1])
                if not (neighbor_as_cell.to_tuple() in explored.keys()):
                    explored[neighbor_as_cell.to_tuple()] = neighbor_as_cell
                    frontier.append(neighbor_as_cell.to_tuple())

    # create closest path
    end_cell_as_tuple = maze.waypoints[0]
    if end_cell_as_tuple in explored.keys():
        curr_cell = explored.get(end_cell_as_tuple, None)
        while not (curr_cell == None):
            closest_path.append(curr_cell.to_tuple())
            next_cell_as_tuple = (curr_cell.prev_row, curr_cell.prev_col)
            curr_cell = explored.get(next_cell_as_tuple, None)
    
    closest_path.reverse()
    return closest_path

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    waypoint_as_tuple = maze.waypoints[0]
    closest_path = []

    first_cell = CellNode(maze.start[0], maze.start[1], -1, -1, 0)
    explored = {first_cell.to_tuple() : first_cell}
    frontier = [(0, first_cell.to_tuple())]
    heapq.heapify(frontier)
    
    # go thru all paths
    while len(frontier):
        curr_cell_as_tuple = heapq.heappop(frontier)[1]
        curr_cell_maze = maze[curr_cell_as_tuple]
        if curr_cell_maze == maze.legend.waypoint:
            break
        else:
            # add neighbors if not at waypoint
            for neighbor in maze.neighbors(*curr_cell_as_tuple):
                neighbor_as_cell = CellNode(neighbor[0], neighbor[1], curr_cell_as_tuple[0], curr_cell_as_tuple[1], explored[curr_cell_as_tuple].dist + 1)
                if not (neighbor_as_cell.to_tuple() in explored.keys()):
                    neighbor_as_tuple = neighbor_as_cell.to_tuple()

                    f_n = neighbor_as_cell.dist + manhattan_distance(waypoint_as_tuple, neighbor_as_tuple)
                    explored[neighbor_as_tuple] = neighbor_as_cell
                    heapq.heappush(frontier, (f_n, neighbor_as_tuple))

    # create closest path
    end_cell_as_tuple = maze.waypoints[0]
    if end_cell_as_tuple in explored.keys():
        curr_cell = explored.get(end_cell_as_tuple, None)
        while not (curr_cell == None):
            closest_path.append(curr_cell.to_tuple())
            next_cell_as_tuple = (curr_cell.prev_row, curr_cell.prev_col)
            curr_cell = explored.get(next_cell_as_tuple, None)
    
    closest_path.reverse()
    return closest_path

def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    closest_path = []
    subpath = []
    all_waypoints = dict.fromkeys(maze.waypoints, False)
    num_waypoints_to_visit = len(all_waypoints.keys())

    first_cell = CellNode(maze.start[0], maze.start[1], -1, -1, 0)
    explored = {first_cell.to_tuple() : first_cell}
    frontier = [(0, first_cell.to_tuple())]
    heapq.heapify(frontier)
    prev_cell_as_tuple = (first_cell.to_tuple()[0] + 1, first_cell.to_tuple()[1])
    
    # go thru all paths
    while len(frontier) > 0:
        curr_cell_as_tuple = heapq.heappop(frontier)[1]
        if not are_neighbors(curr_cell_as_tuple, prev_cell_as_tuple):
            continue
        explored[curr_cell_as_tuple].was_explored = True
        subpath.append(curr_cell_as_tuple)
        curr_cell_maze = maze[curr_cell_as_tuple]
        # check cell for waypoint and update waypoints to visit and closest_path
        if curr_cell_maze == maze.legend.waypoint and all_waypoints[curr_cell_as_tuple] == False:
            all_waypoints[curr_cell_as_tuple] = True
            num_waypoints_to_visit -= 1

            current_waypoint = curr_cell_as_tuple
            closest_path.extend(subpath)
            subpath.clear()
            
            for point in explored.keys():
                explored[point].was_explored = False
            if num_waypoints_to_visit == 0:
                break
            frontier.clear()
        # add neighbors
        mst = MST(get_unvisited_waypoints(all_waypoints))
        for neighbor in maze.neighbors(*curr_cell_as_tuple):
            neighbor_as_tuple = tuple((neighbor[0], neighbor[1]))
            if not (neighbor_as_tuple in explored.keys()):
                neighbor_as_cell = CellNode(neighbor[0], neighbor[1], curr_cell_as_tuple[0], curr_cell_as_tuple[1], explored[curr_cell_as_tuple].dist + 1)
                h_n = get_distance_to_nearest_waypoint(neighbor_as_tuple, all_waypoints) + mst.compute_mst_weight()
                f_n = h_n
                explored[neighbor_as_tuple] = neighbor_as_cell
                heapq.heappush(frontier, (f_n, neighbor_as_tuple))
            elif not (explored[neighbor_as_tuple].was_explored):
                explored[neighbor_as_tuple].prev_row = curr_cell_as_tuple[0]
                explored[neighbor_as_tuple].prev_col = curr_cell_as_tuple[1]
                h_n = get_distance_to_nearest_waypoint(neighbor_as_tuple, all_waypoints) + mst.compute_mst_weight()
                f_n = h_n
                explored[neighbor_as_tuple] = neighbor_as_cell
                heapq.heappush(frontier, (f_n, neighbor_as_tuple))
        prev_cell_as_tuple = curr_cell_as_tuple

    return closest_path

def are_neighbors(a, b):
    return abs(a[0] - b[0]) == 1 or abs(a[1] - b[1]) == 1

#@lru_cache(maxsize=32)
def get_distance_to_nearest_waypoint(here_as_tuple, waypoints_dict):
    unvisited_waypoints = get_unvisited_waypoints(waypoints_dict)
    min_distance = manhattan_distance(here_as_tuple, unvisited_waypoints[0])
    for waypoint_tuple in unvisited_waypoints:
        new_distance = manhattan_distance(waypoint_tuple, here_as_tuple)
        if new_distance < min_distance:
            min_distance = new_distance
    return min_distance

#@lru_cache(maxsize=4)
def get_unvisited_waypoints(waypoints_dict):
    return [k for k,v in waypoints_dict.items() if v == False]

def fast(maze):
    """
    Runs suboptimal search algorithm for extra credit/part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    return []


