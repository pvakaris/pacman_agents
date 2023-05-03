# parsons/20-nov-2017
#
# Version 1
#
# The starting point for CW2.
#
# Intended to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agent here is was written by Simon Parsons, based on the code in
# pacmanAgents.py

from pacman import Directions
from game import Agent
import api
import random
import game
import util
from map import PacmanMap, WalkableCell, WallCell, Cell, Config


# Pacman agent that wins games using MDP solver
class MDPAgent(Agent):

    # Constructor: this gets run when we first invoke pacman.py
    def __init__(self):
        self.map = None

    # Gets run after an MDPAgent object is created and once there is
    # game state to access.
    def registerInitialState(self, state):
        if self.map is None:
            self.map = PacmanMap(state)
        
    # This is what gets run in between multiple games
    def final(self, state):
        print("Looks like the game just ended!")

    # Run at every tick of the game
    # Updates the map representation according to the newest state
    # Runs ValueIteration to calculate the utility value of each state
    # Chooses the action to take according to the optimalPolicy
    def getAction(self, state):
        self.map.update(state)
        legal = api.legalActions(state)
        my_coordinates = api.whereAmI(state)
        # self.map.printMap(my_coordinates)
        # Run the algorithm specified in the Config class "run_until_until_convergence"
        if(Config.run_until_until_convergence):
            self.runValueIterationUntilConvergence()
        else:
            self.runValueIterationForLimitedCycles()
            
        # self.map.printUtilities(my_coordinates)
        self.removeStop(legal)
        best_move = self.getOptimalPolicy(my_coordinates, legal)
        self.map.overrideUtilities()
        
        return api.makeMove(best_move, legal)
    
    # Value Iteration algorithm that runs for a limited amount of iterations and updates the utility values of all WalkableCells in the map
    def runValueIterationForLimitedCycles(self):
        for i in range(Config.number_of_iterations):
            new_utilities = self.getNewUtilities()
            # Set new utilities for each WalkableCell
            for (coordinates, new_utility) in new_utilities:
                self.map.getCell(coordinates).utility = new_utility
    
    # Value Iteration algorithm that runs until all utilities converge and updates the utility values of all WalkableCells in the map            
    def runValueIterationUntilConvergence(self):
        previous_utilities = None
        run = True
        while(run):
            new_utilities = self.getNewUtilities()
            run = self.stillRunValueIteration(previous_utilities, new_utilities)
            previous_utilities = new_utilities
            # Set new utilities for each WalkableCell
            for (coordinates, new_utility) in new_utilities:
                self.map.getCell(coordinates).utility = new_utility
    
    # Returns a list of (coordinates, new_utility_value) pairs containing all the walkable cells of the map            
    def getNewUtilities(self):
        height = self.map.height
        width = self.map.width
        new_utilities = []
        for y in range(height):
            for x in range(width):
                coordinates = (x, y)
                if isinstance(self.map.getCell(coordinates), WalkableCell):
                    new_utility = self.getUtility(coordinates)
                    new_utilities.append((coordinates, new_utility))
        return new_utilities
    
    # Used in value iteration algorithm that runs until all the values converge
    # Checks if the difference between previous utility value and current utility value of each
    # state is smalled than the difference needed to stop the algorithm            
    def stillRunValueIteration(self, previous_utilities, new_utilities):
        if(previous_utilities is None): return True
        for x in range(len(previous_utilities)):
            if abs(previous_utilities[x][1] - new_utilities[x][1]) > Config.convergence_difference:
                return True
        return False
            
    # Returns a list with utility values of neighbouring cells (up, right, down, left)        
    def getUtilitiesOfNeighbouringCells(self, coordinates):
        current_utility = self.map.getCell(coordinates).utility
        north_utility = self.checkUtility(self.map.topCell(coordinates), current_utility)
        south_utility = self.checkUtility(self.map.bottomCell(coordinates), current_utility)
        west_utility = self.checkUtility(self.map.leftCell(coordinates), current_utility)
        east_utility = self.checkUtility(self.map.rightCell(coordinates), current_utility)

        utility_up = Config.possibility_straight * north_utility + Config.possibility_left * west_utility + Config.possibility_right * east_utility
        utility_right = Config.possibility_straight * east_utility + Config.possibility_left * north_utility + Config.possibility_right * south_utility
        utility_down = Config.possibility_straight * south_utility + Config.possibility_left * east_utility + Config.possibility_right * west_utility
        utility_left = Config.possibility_straight * west_utility + Config.possibility_left * south_utility + Config.possibility_right * north_utility
        
        return [utility_up, utility_right, utility_down, utility_left]
    
    # Bellman's equation calculating the utility value of the state using utilities of its neighbours 
    def getUtility(self, coordinates):
        current_cell = self.map.getCell(coordinates)
        reward = self.getReward(current_cell)
        # Neighbour utilities
        utilities = self.getUtilitiesOfNeighbouringCells(coordinates)
        return reward + (Config.discount_factor * max((utilities[0], utilities[1], utilities[2], utilities[3])))

    # Having all the utilities of the neighbours, calculates which way is best (which way has the biggest utility)
    def getOptimalPolicy(self, coordinates, legal):
        utilities = self.getUtilitiesOfNeighbouringCells(coordinates)
        actions_and_utilities = []
        
        for action in legal:
            if action == Directions.NORTH:
                actions_and_utilities.append((action, utilities[0]))
            if action == Directions.EAST:
                actions_and_utilities.append((action, utilities[1]))
            if action == Directions.SOUTH:
                actions_and_utilities.append((action, utilities[2]))
            if action == Directions.WEST:
                actions_and_utilities.append((action, utilities[3]))

        return max(actions_and_utilities, key = lambda i : i[1])[0]

    # According to the state of the Cell, return a reward
    def getReward(self, cell):
        if isinstance(cell, WallCell):
            return Config.reward_wall
        else:
            if cell.hasGhost:
                return Config.reward_ghost
            elif self.map.hasGhostClose((cell.x, cell.y)):
                return Config.reward_ghost_neighbour_first
            # elif self.map.neighbourHasGhostClose((cell.x, cell.y)):
            #     return Config.reward_ghost_neighbour_second
            elif cell.hasCapsule:
                return Config.reward_capsule
            elif cell.hasFood:
                return Config.reward_food
            else:
                return Config.reward_empty

    # If the provided cell is not WallCell, returns its utility
    def checkUtility(self, cell, default_utility):
        if isinstance(cell, WallCell):
            return default_utility
        else:
            return cell.utility

    # Removes Directions.STOP from the list of actions
    def removeStop(self, actions):
        # Get the actions we can try, and remove "STOP" if that is one of them.
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)