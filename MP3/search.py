from collections import deque

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
    def __init__(self, objectives):
        self.elements = {key: None for key in objectives}

        # TODO: implement some distance between two objectives
        # ... either compute the shortest path between them, or just use the manhattan distance between the objectives
        self.distances   = {
                (i, j): DISTANCE(i, j)
                for i, j in self.cross(objectives)
            }

    # Prim's algorithm adds edges to the MST in sorted order as long as they don't create a cycle
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
    def __init__(self, row, col, prev_row, prev_col):
        self.row = row
        self.col = col
        self.prev_row = prev_row
        self.prev_col = prev_col
        # self.row_col = str(row) + "," + str(col)
    
    def to_tuple(self):
        return tuple((self.row, self.col))

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
                if maze.navigable(*neighbor) and not (neighbor_as_cell.to_tuple() in explored.keys()):
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
    return []

def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    return []

def fast(maze):
    """
    Runs suboptimal search algorithm for extra credit/part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    return []


