# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """


  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
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

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    ghostPos = successorGameState.getGhostPositions()
    


    #if oldFood[newPos] = true, then +1 bc that means food in next spot
    
    #newScaredTimes = number of moves that each ghost will remain scared bc of eating a power pellet, +newScaredTimes
    
    #newGhostStates is where ghosts are after move, -1(or more) if ghost in next state
    
    #should also give a penalty for not moving
    
    #can see distance to food bc oldFood shows position of all remaining food
    
    foodPos = oldFood.asList()
    distances = []
    nextDistances= []
    
    xPos, yPos = newPos # assign x and y value of new position
    if oldFood[xPos][yPos] == True:
        food = 5; #if there is food, food = 1 for that position
    else:
        food = 0; #if no food, food value = 0
        
    scaredTime = newScaredTimes[0]  # num moves each ghost will remain scared after move
    scaredTotal = 0
    for time in newScaredTimes:
        scaredTotal += time
    scaredAvg = float(scaredTotal)/len(newScaredTimes)
    
    if newPos in ghostPos:
        ghost = -1000000;
    else:
        ghost = 1;
        
    
    for pos in foodPos:
        d = abs(newPos[0] - pos[0]) + abs(newPos[1] - pos[1])
        distances.append(d)
        
        
    v = 100000
    for dis in distances: #find the smallest distance
        if dis < v:
            v = dis
         
    if v == 0:
        disVal = 21
    else:   
        disVal = 20.0/v #gives a larger number to smaller distances
    
            
    if newPos == currentGameState.getPacmanPosition():
        penalty = -5
    else:
        penalty = 0


    
    return food + scaredTime + ghost + penalty + disVal

   # return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
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
   
    """ Problem 2"""
    
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.
    
          Here are some method calls that might be useful when implementing minimax.
    
          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1
            self.index
    
          Directions.STOP:
            The stop direction, which is always legal
    
          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action
    
          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        
        #        if numAgents > 1: #create a list of the agents to iterate over
#            agentList.append(0) #add pacman agents
#            for x in range(numAgents-1): #add the correct amount of ghosts to the list
#                agentList.append(x+1)
#        print agentList
        
       # for a in gameState.getLegalActions(self.index):
       #print gameState.generateSuccessor(self.index,a)
        
        
        #should create a list of agents so i can get their agent index to see if pacman or ghost
        #only decrement depth in max function bc that will only be called once per layer

        #need to return the action from gameState.getLegalActions(agentIndex) that maximizes min-value
        
        depth = self.depth
        #i = 0
        numAgents = gameState.getNumAgents()
        agentList = []

        
        def miniVal(state, d , i):
            if state.isWin() or state.isLose() or d > depth:
                return scoreEvaluationFunction(state)
            v = float("inf")
            actions = state.getLegalActions(i)
            if 'Stop' in actions:
                actions.remove('Stop')
            for a in actions:
                if i+1 >= numAgents: #if it is the last ghost then call max
                    d+=1
                    v = min(v, maxiVal(state.generateSuccessor(i,a), d, 0))
                else: #if it is not the last ghost
                    v = min(v, miniVal(state.generateSuccessor(i,a), d, i+1))
            return v
        
        def maxiVal(state, d, i):
            if state.isWin() or state.isLose() or d > depth:
                return scoreEvaluationFunction(state)
            v = float("-inf")
            actions = state.getLegalActions(i)
            if 'Stop' in actions:
                actions.remove("Stop")
            for a in actions: #pacman
                v = max(v, miniVal(state.generateSuccessor(i,a), d, i+1))
            return v
            
        maxV = float("-inf")
        bestMove = "Stop"
        legalActions = gameState.getLegalActions(self.index)
        legalActions.remove(Directions.STOP)
        
        for a in legalActions:
            val = miniVal(gameState.generateSuccessor(0,a), 1, 1)
            if val > maxV:
                maxV = val
                bestMove = a
     
     
        return bestMove
        
        
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    depth = self.depth
    #i = 0
    numAgents = gameState.getNumAgents()
    actionList = []
    valList = []
    aList = []
    a = float("-inf")
    b = float("inf")
    
    
    
    def maxiVal(state, d, i, alpha, beta):
        if state.isWin() or state.isLose() or d > depth:
            return self.evaluationFunction(state)
            #return scoreEvaluationFunction(state)
        v = float("-inf")
        actions = state.getLegalActions(i)
        if 'Stop' in actions:
            actions.remove("Stop")
        for a in actions: #pacman
            v = max(v, miniVal(state.generateSuccessor(i,a), d, i+1, alpha, beta))
            if v >= beta:
                return v
            alpha = max(v, alpha)
        return v

    def miniVal(state, d , i, alpha, beta):
        if state.isWin() or state.isLose() or d > depth:
            return self.evaluationFunction(state)
            #return scoreEvaluationFunction(state)
        v = float("inf")
        actions = state.getLegalActions(i)
        if 'Stop' in actions:
            actions.remove('Stop')
        for a in actions:
            if i+1 >= numAgents: #if it is the last ghost then call max
                d+=1
                v = min(v, maxiVal(state.generateSuccessor(i,a), d, 0, alpha, beta))
#                if v <= alpha:
#                    return v
#                beta = min(v,beta)
            else: #if it is not the last ghost
                v = min(v, miniVal(state.generateSuccessor(i,a), d, i+1, alpha, beta))
        if v<= alpha:
            return v
        beta = min(v,beta)
        return v
        
    maxV = float("-inf")
#    bestMove = "Stop"
    legalActions = gameState.getLegalActions(self.index)
    legalActions.remove(Directions.STOP)
    
    for a in legalActions:
        val = miniVal(gameState.generateSuccessor(0,a), 1, 1, a, b)
        actionList.append((val, a)) # make a list with all the actions and their values
#        if val >= maxV:
#            maxV = val
#            bestMove = a

    for a in actionList:
        valList.append(a[0]) #create a list of values
        
    maxVal = max(valList)
    for a in actionList:
        if a[0] == maxVal:
            aList.append(a[1])
            
    random.shuffle(aList)
    index = random.randint(0, len(aList)-1)

    return aList[index]





    util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    
    depth = self.depth
    numAgents = gameState.getNumAgents()
    #agentList = []
    
    def expectation(vals,weights):
        expectation = 0
        for i in range(len(vals)):
            expectation += (vals[i]*weights[i])
        return expectation
        

    
    def expectiVal(state, d , i):
        values = []
        weights = []
        if state.isWin() or state.isLose() or d > depth:
            return self.evaluationFunction(state)
            #return scoreEvaluationFunction(state)
        #v = float("inf")
        actions = state.getLegalActions(i)
        if 'Stop' in actions:
            actions.remove('Stop')
        for a in actions:
            if i+1 >= numAgents: #if it is the last ghost then call max
                d+=1
                values.append(maxiVal(state.generateSuccessor(i,a), d, 0))
                weights.append(1.0/len(actions))
                #v = min(v, maxiVal(state.generateSuccessor(i,a), d, 0))
            else: #if it is not the last ghost
                values.append(expectiVal(state.generateSuccessor(i,a), d, i+1))
                weights.append(1.0/len(actions))
                #v = min(v, miniVal(state.generateSuccessor(i,a), d, i+1))
        return expectation(values, weights)
    
    def maxiVal(state, d, i):
        if state.isWin() or state.isLose() or d > depth:
            return self.evaluationFunction(state)
            #return scoreEvaluationFunction(state)
        v = float("-inf")
        actions = state.getLegalActions(i)
        if 'Stop' in actions:
            actions.remove("Stop")
        for a in actions: #pacman
            v = max(v, expectiVal(state.generateSuccessor(i,a), d, i+1))
        return v
        
    maxV = float("-inf")
    bestMove = "Stop"
    legalActions = gameState.getLegalActions(self.index)
    legalActions.remove(Directions.STOP)
    
    for a in legalActions:
        val = expectiVal(gameState.generateSuccessor(0,a), 1, 1)
        if val > maxV:
            maxV = val
            bestMove = a
 
    return bestMove
    
    util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Awards pacman if there is food in the current state, awards if state is a win, punishes if state is a lose, and neutral
    if neither. Punishes if there is a ghost in the same position. Gives the average of the scared times of the ghosts, so a higher
    average is better. Rewards pacman for no remaining food, or else gives a value proportional to the amount of food left
    on the board. Checks to see how many of its successors have food, and adds 1 for each successor with food. 
  """

  
  successors = []
  succFoodPos = []
  
  gameFood = currentGameState.getFood() # all the positions of the remaining food
  pacPos = currentGameState.getPacmanPosition() # curr position of pacman
  win = currentGameState.isWin() #check to see if curr game state is a win
  lose = currentGameState.isLose() #checks if a loss    
  legalActions = currentGameState.getLegalPacmanActions() 
  ghostStates = currentGameState.getGhostStates() # current ghost states
  ghostPos = currentGameState.getGhostPositions() # will use this to check this state against a ghost state, which will = a lose
  foodLeft = currentGameState.getNumFood() # will give larger score if this state results in less food left on the board
  scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates] #number of moves that each ghost will remain scared
  scoreEval = scoreEvaluationFunction(currentGameState) #the eval function of the state
  
  
#Food Check#
  xPos, yPos = pacPos # assign x and y value of new position
  if gameFood[xPos][yPos] == True:
      food = 10; #if there is food, food = 10 for that position
  else:
      food = -5; #if no food, penelized for no food
      
      
#Win check# 
  if win == True:
      win = 100
  elif lose == True:
      win = -100
  else:
      win = 0
  
#Ghost check#
  if pacPos in ghostPos:
      pos = -100
  else:
      pos = 5
  
  
#Scared check#
  scaredTotal = 0
  for time in scaredTimes:
      scaredTotal += time
  scaredAvg = float(scaredTotal)/len(scaredTimes)
  
  
#Food left#  
  remainingFood = 0
  if foodLeft == 0:
      remainingFood = 100
  else:
      remainingFood = 500/foodLeft
  
#Check if successors of this state have food#
  totalSuccFood = 0
  for a in legalActions:
      successors.append(currentGameState.generatePacmanSuccessor(a))
  for succ in successors:
      succFoodPos.append(succ.getPacmanPosition())
  for succ in succFoodPos:
      xP, yP = succ
      if gameFood[xP][yP] == True:
          totalSuccFood += 10 #add 2 for each successor that has food in it
  
  
  
  
  
  return food + win + pos + scaredAvg + scoreEval + remainingFood + totalSuccFood 
  #check
  
  
  
  
  
      
  
  


    
      
  
  
  
  
  
  
  
  
  
  
  
  util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

