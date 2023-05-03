import api 

class Config:
    discount_factor = 0.9
    possibility_straight = 0.8
    possibility_left = 0.1
    possibility_right = 0.1
    reward_empty = 5.0
    reward_wall = -10.0
    reward_food = 10.0
    reward_capsule = 5.0
    reward_ghost = -1000.0
    # Neighbouring cell has a ghost
    reward_ghost_neighbour_first = -500.0
    # Neighbour of the neighbouring cell has a ghost
    reward_ghost_neighbour_second = -250.0
    number_of_iterations = 100
    # Set to True if run value iteration until full convergence, otherwise the iteration will be run "number_of_iterations" times
    run_until_until_convergence = False
    # When the difference between iterations is smaller than this amount, finish the value iteration. The values have now converged
    convergence_difference = 0.01

# Representation of the map that is stored in the Pacman's "head"
class PacmanMap:
    def __init__(self, state):
        self.height = 0
        self.width = 0
        self.grid = self.initGrid(state)
        self.cellsWithGhosts = []
        self.cellsWithCapsules = []
        self.ghostPrevious = []
        self.draw(state)
        self.fill(state)

    # Updates the grid according to the newest state
    def update(self, state):
        cell = self.getCell(api.whereAmI(state))
        self.markCapsules(cell)
        cell.hasFood = False
        cell.hasCapsule = False
        self.markGhosts(state)

    # Gives a Cell in the specified position
    def getCell(self, position):
        return self.grid[position[1]][position[0]]    
    
    # Gives a Cell above the specified position
    def topCell(self, position):
        return self.grid[position[1] + 1][position[0]]

    # Gives a Cell below the specified position
    def bottomCell(self, position):
        return self.grid[position[1] - 1][position[0]]

    # Gives a Cell right of the specified position
    def leftCell(self, position):
        return self.grid[position[1]][position[0] - 1]

    # Gives a Cell right of the specified position
    def rightCell(self, position):
        return self.grid[position[1]][position[0] + 1]

    # Update the list containing capsules
    def markCapsules(self, currentCell):
        if(currentCell.hasCapsule):
            self.cellsWithCapsules.remove(currentCell)

    # Checks in neighbouring WalkableCells have ghosts
    def hasGhostClose(self, coordinates):
        if coordinates[1] + 1 < self.height:
            top = self.topCell(coordinates)
            if(isinstance(top, WalkableCell)):
                if top.hasGhost:
                    return True
        if coordinates[1] - 1 >= 0:
            bottom = self.bottomCell(coordinates)
            if(isinstance(bottom, WalkableCell)):
                if bottom.hasGhost:
                    return True
        if coordinates[0] - 1 >= 0:        
            left = self.leftCell(coordinates)
            if(isinstance(left, WalkableCell)):
                if left.hasGhost:
                    return True
        if coordinates[0] + 1 < self.width:
            right = self.rightCell(coordinates)
            if(isinstance(right, WalkableCell)):
                if right.hasGhost:
                    return True
        return False
    
    # Checks if the neighbouring cells of the neighbouring cells have ghosts
    def neighbourHasGhostClose(self, coordinates):
        if self.hasGhostClose((coordinates[0] + 1, coordinates[1])) or self.hasGhostClose((coordinates[0] - 1, coordinates[1])) or self.hasGhostClose((coordinates[0], coordinates[1] + 1)) or self.hasGhostClose((coordinates[0], coordinates[1] - 1)):
            return True
        else:
            return False

    # Marks which cells ghosts currently occupy
    def markGhosts(self, state):
        # Remoes ghosts from all Cells
        self.ghostPrevious = []
        for cell in self.cellsWithGhosts:
            cell.hasGhost = False
            self.ghostPrevious.append(cell)

        # Marks them again using the newest state (if ghosts qchanged the position)
        self.cellsWithGhosts = []
        ghosts = api.ghosts(state)
        for ghost in ghosts:
            y = int(ghost[1])
            x = int(ghost[0])
            cell = self.grid[y][x]
            cell.hasGhost = True
            self.cellsWithGhosts.append(cell)
        
    # Makes the utilities of all the WalkableCells in the map 0.0    
    def overrideUtilities(self):
        for y in range(self.height):
            for x in range(self.width):
                coordinates = (x, y)
                if isinstance(self.getCell(coordinates), WalkableCell):
                    self.getCell(coordinates).utility = 0.0

    # =================================
    # Creation methods called only once
    # =================================

    # Fills the grid when it is first initialised
    def fill(self, state):
        self.fillWithFood(state)
        self.fillWithCapsules(state)
        self.markGhosts(state)

    # Determines which of the WalkableCells have food in them
    def fillWithFood(self, state):
        foodGrid = state.getFood()
        for y in range(self.height):
            for x in range(self.width):
                if foodGrid[x][y] == True:
                    self.grid[y][x].hasFood = True

    # Determines which of the WalkableCells have capsules in them
    def fillWithCapsules(self, state):
        capsules = api.capsules(state)
        for capsule in capsules:
            y = capsule[1]
            x = capsule[0]
            cell = self.grid[y][x]
            cell.hasCapsule = True
            cell.hasFood = False
            self.cellsWithCapsules.append(cell)

    # Creates a 2D matrix with all the positions as None and assigns to the grid variable
    def initGrid(self, state):
        wallGrid = state.getWalls()
        self.width = wallGrid.width
        self.height = wallGrid.height
        grid = [[None for x in range(self.width)] for y in range(self.height)]
        return grid

    # Each cell of the matrix is assigned either a WallCell or a WalkableCell
    def draw(self, state):
        wallGrid = state.getWalls()

        # Mark walls and walkable cells
        for x in range(self.width):
            for y in range(self.height):
                if wallGrid[x][y] == True:
                    self.grid[y][x] = WallCell(x, y)
                else:
                    self.grid[y][x] = WalkableCell(x, y)
                    
    # =================================
    # Printing methods
    # =================================
                    
    # Print the map to the terminal
    def printMap(self, pacman_coordinates):
        print("\n\n\n")
        for y in reversed(range(self.height)):
            for x in range(self.width):
                if isinstance(self.grid[y][x], WallCell):
                    print("#"),
                else:
                    if y == pacman_coordinates[1] and x == pacman_coordinates[0]:
                        print("M"),
                    elif self.grid[y][x].hasCapsule:
                        print("o"),
                    elif self.grid[y][x].hasGhost:
                        print("G"),
                    elif self.grid[y][x].hasFood:
                        print("."),
                    else:
                        print("_"),
            print("")
        print("\n\n\n")
        
    # Print the map with utility values
    def printUtilities(self, pacman_coordinates):
        print("\n\n\n")
        for y in reversed(range(self.height)):
            for x in range(self.width):
                if isinstance(self.grid[y][x], WallCell):
                    print(" ## "),
                else:
                    if y == pacman_coordinates[1] and x == pacman_coordinates[0]:
                        print("MMM"),
                    elif self.getCell((x, y)).hasGhost:
                        print("!" + str("%.1f" % self.grid[y][x].utility)),
                    else:
                        print(str("%.2f" % self.grid[y][x].utility)),
            print("")
        print("\n\n\n")


# What the map consists of
class Cell(object):
    def __init__(self, xCoordinate, yCoordinate):
        self.x = xCoordinate
        self.y = yCoordinate


# A subclass of Cell which indicates that Pacman can stand in this Cell
class WalkableCell(Cell):
    def __init__(self, x, y, hasFood = False, hasCapsule = False, hasGhost = False, utility = 0.0):
        super(WalkableCell, self).__init__(x, y)
        self.hasFood = hasFood
        self.hasCapsule = hasCapsule
        self.hasGhost = hasGhost
        self.utility = utility


# A subclass of Cell which indicates that Pacman cannot enter this Cell
class WallCell(Cell):
    def __init__(self, x, y):
        super(WallCell, self).__init__(x, y)

