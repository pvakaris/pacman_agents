# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
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

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

# Commands that will be run in evaluation:
#   - python3 pacman.py -p Q3LearnAgent -x 2000 -n 2010 -l smallGrid
#   - python3 pacman.py -p QLearnAgent -x 2000 -n 2010 -l *someRandomGrid*

from __future__ import absolute_import
from __future__ import print_function

import random

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util
from queue import PriorityQueue

# GameStateFeatures should only contain info about the game i think (the qvalues and visitations should be held in the agent)
# fix the q values dictionary / ~ ~ maybe combine it with the visitations ~ ~
#

class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm
    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """
        # ADDITIONAL STUFF CAN BE ADDED HERE

        self.state = state

        # Might come in handy to store information about the last decision
        self.previous_states = []
        self.previous_actions = []
    
    def getLegalActions(self):
        return self.state.getLegalPacmanActions()

    # A* search using priority queue
    # Finds shortest distance between pacman and an object
    def shortestDistance(self, pacman, object, walls):

        # priority queue that stores the next positions to be explored in the A* search algorithm
        frontier = PriorityQueue()

        # dict that maps a position to the previous position visited in the search, forming the shortest path
        came_from = {}

        # dict that maps a position to the cost (distance) to reach that position from the starting position (pacman)
        cost_so_far = {}

        # Init pacman as starting position
        frontier.put((0, pacman))
        came_from[pacman] = None
        cost_so_far[pacman] = 0

        # Loops until object coordinates are found
        while not frontier.empty():
            _, current = frontier.get()

            if current == object:
                break
            
            # Searches for object by incrementing by one the coordinates in every direction starting from starting position (pacman)
            for next_pos in [(current[0] + 1, current[1]), (current[0] - 1, current[1]),
                            (current[0], current[1] + 1), (current[0], current[1] - 1)]:
                
                # Check if next postion is a wall
                if next_pos in walls:
                    continue
                
                new_cost = cost_so_far[current] + 1
                # Check if next position hasn't been visited or cost to reach said position is inferior
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    # Update cost
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self.estimateDistance(object, next_pos)
                    frontier.put((priority, next_pos))
                    came_from[next_pos] = current

        # Compute the distance between pacman and object
        distance = 0
        current = object

        # Increments distance for every step between the object and pacman
        # Until it reaches pacman
        while current != pacman:
    
            distance += 1

            # Exception catcher to resolve error where scared ghosts have position of (x +/-O.5, y +/-0.5)
            try:
                current = came_from[current]
            except Exception:
               # If coordinates float turn to int
               current = (round(current[0]),round(current[1]))

        return distance

    # computes the estimated distance between two positions
    def estimateDistance(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])


class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.2,
                 epsilon: float = 0.05,
                 gamma: float = 0.8,
                 maxAttempts: int = 30,
                 numTraining: int = 300):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.
        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.
        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0

        # Set epsilon to 0.1 insted of default 0.05
        self.setEpsilon(0.1)

        # Where the Q-value table is
        self.q_values = util.Counter()

        # Where the visitations are stored --> used for frequencies
        self.visitations = util.Counter()
    
    # =============================================================================================================================
    # ===========================================================[ Helpers ]=======================================================
    # =============================================================================================================================

    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts
    
    # =============================================================================================================================
    # =============================================================================================================================
    # =============================================================================================================================

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state
        Returns:
            The reward assigned for the given trajectory
        """
        "*** YOUR CODE HERE ***"

        # Checks if the game is SmallGrid or not
        if startState.getWalls().width == 7 and startState.getWalls().height == 7:
            #Values to compute possible reward
            enemyReward = -10
            foodReward = 7
            empty = -0.5
            capsuleReward = 5

            walls = startState.getWalls().asList()
            pacmanNextPosition = endState.getPacmanPosition()
            
            reward = 0

            # Assigne respective rewards based on pacman's trajectory
            if pacmanNextPosition in startState.getGhostPositions():
                reward = enemyReward
            elif pacmanNextPosition in startState.getFood().asList():
                reward =  foodReward
            elif pacmanNextPosition in startState.getCapsules():
                reward = capsuleReward
            else:
                reward = empty


            ghosts = startState.getGhostPositions()
            # Checks if a ghost is close to pacman 
            # The closer a ghost is the more negative a reward is
            for ghost in ghosts:
                distanceFromGhost = GameStateFeatures(endState).shortestDistance(pacmanNextPosition, ghost, walls)
                if distanceFromGhost <= 4:
                    reward = reward - 1
                else:
                    reward = reward + 0.5

            foods = startState.getFood().asList()
            # Checks if a food is close to pacman 
            # The closer a food is the more positive a reward is
            for food in foods:
                distanceFromFood = GameStateFeatures(startState).shortestDistance(pacmanNextPosition, food, walls)
                if distanceFromFood <= 1:
                    reward = reward + 1
            
            return reward
        # If the game isn't a small grid score differiential between before and after pacman's trajectory is used
        else:
            
            return endState.getScore() - startState.getScore()
        
    
    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take
        Returns:
            Q(state, action)
        """
        return self.q_values[(state, action)]


    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state
        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        # Initialise an empty list
        q_values = []

        # Look through all legal actions and return the one with highest known utility given the state
        legal_actions = state.getLegalActions()
        for action in legal_actions:
            if self.visitations[(state,action)] != 0:
                q = self.getQValue(state, action)
                q_values.append(q)
        # If utilities aren't found initialize to 0
        if len(q_values) == 0:
            return 0 
        else:
            return max(q_values)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Performs a Q-learning update
        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """

        # Ref: https://keats.kcl.ac.uk/pluginfile.php/8500678/mod_resource/content/15/rl2.pdf (slide 42)
        self.q_values[(state, action)] = self.getQValue(state, action) + self.alpha * (reward + self.gamma * self.maxQValue(nextState) - self.getQValue(state,action))


    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.
        Args:
            state: Starting state
            action: Action taken
        """
        # Increment by one if an action in that state is made
        self.visitations[(state, action)] += 1

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken
        Returns:
            Number of times that the action has been taken in a given state
        """

        return self.visitations[(state, action)]

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def explorationFn(self) -> float:
        """
        Computes exploration function.
        Return a value based on the counts
        HINT: Do a greed-pick or a least-pick
        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited
        Returns:
            The exploration value
        """

        return (random.randint(0,100)/100)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning
        If you wish to use epsilon-greedy exploration, implement it in this method.
        HINT: look at pacman_utils.util.flipCoin
        Args:
            state: the current state
        Returns:
            The action to take
        """
        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
    
        # Dictionary containing all utlities of states that can be visited
        actionUtility = {}
        for possibleAction in legal:
            actionUtility[possibleAction] = self.getQValue(state, possibleAction)
        # returns state with max utility out of all legal states
        bestAction = max(actionUtility, key=actionUtility.get)

        # Is the agent randomly exploring the field or choosing the best action at the time (Epsilon-greedy)
        if self.explorationFn() < self.epsilon:
            chosenAction = random.choice(legal)
        else:
            chosenAction = bestAction

        # Removes entries where utilities are 0 (created by the Counter class)
        for k,v in list(self.q_values.items()):
            if v == 0:
                del self.q_values[k]

        # Update Q-values

        nextState = state.generatePacmanSuccessor(chosenAction)
        reward = self.computeReward(state, nextState)
        self.learn(state, chosenAction, reward, nextState)

        self.updateCount(state, chosenAction)
       
        # Returns best action or possibly random chosen action when being trained
        return chosenAction



    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.
        Args:
            state: the final game state
        """
        print(f"Game {self.getEpisodesSoFar()} just ended!")
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)