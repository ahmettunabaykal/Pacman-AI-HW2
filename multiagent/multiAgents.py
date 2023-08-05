# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
       
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newCapsules = successorGameState.getCapsules()
        newGhostStates = successorGameState.getGhostStates()
       
        from util import manhattanDistance

        foodDistances = [manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]
        nearestFood = min(foodDistances) if foodDistances else 0
        remainingFood = len(newFood.asList())
        remainingPowerPellets = len(newCapsules)
        ghostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        nearestGhostDistance = min(ghostDistances)
        scaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        if newPos in [ghost.getPosition() for ghost in newGhostStates]:
            return float('-inf')

        evaluationScore = successorGameState.getScore()

        if scaredTimes:
            bonus = sum(scaredTimes) / len(scaredTimes)
            evaluationScore += bonus

        if nearestFood:
            bonus = 1.0 / nearestFood
            evaluationScore += bonus

        if remainingFood:
            bonus = remainingFood / 100.0
            evaluationScore += bonus

        if remainingPowerPellets:
            bonus = remainingPowerPellets / 50.0
            evaluationScore += bonus

        if nearestGhostDistance <= 2:
            penalty = nearestGhostDistance / 10.0
            evaluationScore -= penalty

        return evaluationScore

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimax(gameState, depth, agent):
            
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState), None

            actions = gameState.getLegalActions(agent)
            successors = [gameState.generateSuccessor(agent, action) for action in actions]
            if agent == 0:
                values = [minimax(successor, depth, agent + 1)[0] for successor in successors]
                max_value = max(values)
                max_indices = [index for index in range(len(values)) if values[index] == max_value]
                return max_value, actions[random.choice(max_indices)]
            else:
                next_agent = agent + 1
                if next_agent == gameState.getNumAgents():
                    next_agent = 0
                if next_agent == 0:
                    depth += 1
                values = [minimax(successor, depth, next_agent)[0] for successor in successors]
                min_value = min(values)
                min_indices = [index for index in range(len(values)) if values[index] == min_value]
                return min_value, actions[random.choice(min_indices)]

        return minimax(gameState, 0, 0)[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        

        def max_value(state, depth, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state), None

            v = float('-inf')
            best_action = None
            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)
                value = min_value(successor, depth, 1, alpha, beta)[0]
                if value > v:
                    v = value
                    best_action = action
                if v > beta:
                    return v, best_action
                alpha = max(alpha, v)
            return v, best_action

        def min_value(state, depth, agent, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state), None

            v = float('inf')
            best_action = None
            for action in state.getLegalActions(agent):
                successor = state.generateSuccessor(agent, action)
                if agent == state.getNumAgents() - 1:
                    value = max_value(successor, depth + 1, alpha, beta)[0]
                else:
                    value = min_value(successor, depth, agent + 1, alpha, beta)[0]
                if value < v:
                    v = value
                    best_action = action
                if v < alpha:
                    return v, best_action
                beta = min(beta, v)
            return v, best_action

        return max_value(gameState, 0, float('-inf'), float('inf'))[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        
        def max_value(gameState, depth):
            max_score = float("-inf")
            max_action = None
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState), None
            for action in gameState.getLegalActions(0):
                successor = gameState.generateSuccessor(0, action)
                score = exp_value(successor, depth, 1)
                if score > max_score:
                    max_score = score
                    max_action = action
            return max_score, max_action

        def exp_value(gameState, depth, ghost_index):
            total_score = 0
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            actions = gameState.getLegalActions(ghost_index)
            if ghost_index == gameState.getNumAgents() - 1:
                for action in actions:
                    successor = gameState.generateSuccessor(ghost_index, action)
                    total_score += max_value(successor, depth + 1)[0]
            else:
                for action in actions:
                    successor = gameState.generateSuccessor(ghost_index, action)
                    total_score += exp_value(successor, depth, ghost_index + 1)
            return total_score / len(actions)

        return max_value(gameState, 0)[1]


def betterEvaluationFunction(currentGameState: GameState) -> float:
    """
    First gets information about the game state
    Then, computes the nearst ghost and food and the capsule
    Then, computes the number of remaining food and capsules
    After that computes, if the game is win or deadend or lost
    Lastly , it computes the score according to situations stated above.
    """

    # Extract useful information from the current state
    pacmanPos = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    capsules = currentGameState.getCapsules()
    ghosts = currentGameState.getGhostStates()
    score = currentGameState.getScore()

    # Compute the distances to the nearest food pellet, power capsule, and ghost
    from util import manhattanDistance

    foodDistances = [manhattanDistance(pacmanPos, foodPos) for foodPos in foodGrid.asList()]
    nearestFood = min(foodDistances) if foodDistances else 0
    capsuleDistances = [manhattanDistance(pacmanPos, capsulePos) for capsulePos in capsules]
    nearestCapsule = min(capsuleDistances) if capsuleDistances else 0
    ghostDistances = [manhattanDistance(pacmanPos, ghost.getPosition()) for ghost in ghosts]
    nearestGhost = min(ghostDistances) if ghostDistances else 0

    # Compute the remaining food count
    remainingFood = foodGrid.count()

    # Compute if Pac-Man is in a dead-end
    actions = currentGameState.getLegalPacmanActions()
    reverseAction = Directions.REVERSE[currentGameState.getPacmanState().getDirection()]
    if len(actions) == 1 and actions[0] == reverseAction:
        deadEnd = 1
    else:
        deadEnd = 0

    # Compute the score based on the factors mentioned above
    evaluationScore = score
    if nearestFood:
        evaluationScore += 1.0 / nearestFood
    if remainingFood:
        evaluationScore += remainingFood / 100.0
    if nearestCapsule:
        evaluationScore += 1.0 / nearestCapsule
    if nearestGhost <= 1:
        evaluationScore -= 1000.0
    elif nearestGhost <= 3:
        evaluationScore -= 100.0
    elif nearestGhost <= 5:
        evaluationScore -= 10.0
    if currentGameState.isLose():
        evaluationScore -= 10000.0
    if currentGameState.isWin():
        evaluationScore += 10000.0
    if deadEnd:
        evaluationScore -= 100.0

    return evaluationScore

# Abbreviation
better = betterEvaluationFunction
