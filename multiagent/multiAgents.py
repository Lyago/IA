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
    ghostPositions = [ghost.getPosition() for ghost in newGhostStates]
    "*** YOUR CODE HERE ***"
    #Scores relevantes pro algoritmo
    foodScore = 0
    ghostScore = 0
    #distancia da comida mais proxima, usando distancia manhattan
    distanceToClosestFood = min([util.manhattanDistance(newPos, food) for food in oldFood.asList()])
    #distancia do fantasma mais proximo, usando distancia manhattan
    distanceToClosestGhost = min([util.manhattanDistance(newPos, position) for position in ghostPositions])

    #aproximar-se dos fantasmas atribui um score negativo ao agente, aproximar-se da comida atribui um score positivo
    if distanceToClosestGhost == 0:      
      return -99
    elif distanceToClosestGhost < 6:
      ghostScore = (1./distanceToClosestGhost) * -2.0

    if distanceToClosestFood == 0:
      foodScore = 0
      ghostScore += 2
    else:
      foodScore = 1./distanceToClosestFood

    return foodScore + ghostScore


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
  """
    Your minimax agent (question 2)
  """

  def gameOver(self, gameState, d):
    return gameState.isLose() or gameState.isWin() or d == 0

  def minimax(self, gameState, agentIndex, depth):
    legalActions = gameState.getLegalActions(agentIndex)
    if agentIndex == 0 and Directions.STOP in legalActions:
      legalActions.remove(Directions.STOP)
     

    successorStates = [gameState.generateSuccessor(agentIndex, action) for action in legalActions]
    
    if self.gameOver(gameState, depth):
      return self.evaluationFunction(gameState)
    else:
      #o modulo serve para conseguir voltar ao indice 0
      nextAgent = (agentIndex + 1) % gameState.getNumAgents()
      values = [self.minimax(state, nextAgent, depth -1) for state in successorStates]            
      if nextAgent == 0: # pacman
        #print max(values)
        return max(values)
      else:
        #print min(values)
        return min(values)

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    depth = gameState.getNumAgents()*self.depth

    legalActions = gameState.getLegalActions(0)
    legalActions.remove(Directions.STOP)
    
    successorStates = [gameState.generateSuccessor(0, action) for action in legalActions]
    # valores para cada estado sucessor
    values = [self.minimax(state, 1, depth - 1) for state in successorStates]  
             
    # retorna a acao com maior valor associado
    return legalActions[values.index(max(values))]

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """
  def gameOver(self, gameState, d):
    return gameState.isLose() or gameState.isWin() or d == 0

  def alphabeta(self, gameState, agentIndex, depth, alpha, beta):
    legalActions = gameState.getLegalActions(agentIndex)
    if agentIndex == 0 and Directions.STOP in legalActions:
      legalActions.remove(Directions.STOP)
     

    successorStates = [gameState.generateSuccessor(agentIndex, action) for action in legalActions]
    
    if self.gameOver(gameState, depth):
      return self.evaluationFunction(gameState)
    else:
           
      if agentIndex == 0: # pacman
        value = float("inf") 
        for state in successorStates:
          nextAgent = (agentIndex + 1) % gameState.getNumAgents()
          value = max(self.alphabeta(state, nextAgent, depth-1, alpha, beta), value)
          if value >= beta:
            return value
          alpha = max(alpha, value)
        return value
      else:
        value = float("inf")
        for state in successorStates:
          nextAgent = (agentIndex + 1) % gameState.getNumAgents()
          value = max(self.alphabeta(state, nextAgent, depth-1, alpha, beta), value)
          if value <= alpha:
            return value
          beta = min(beta, value)
        return value

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    depth = gameState.getNumAgents()*self.depth

    legalActions = gameState.getLegalActions(0)
    legalActions.remove(Directions.STOP)
    
    successorStates = [gameState.generateSuccessor(0, action) for action in legalActions]
    # valores para cada estado sucessor
    values = [self.alphabeta(state, 1, depth - 1, -1e308, 1e308) for state in successorStates]  
             
    # retorna a acao com maior valor associado
    return legalActions[values.index(max(values))]


class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """ 
  def gameOver(self, gameState, d):
    return gameState.isLose() or gameState.isWin() or d == 0

  def expectimax(self, gameState, agentIndex,  depth):
    """
    Same as minimax, except we do an average of min. 
    We do an average because the ghost behavior is expected to be 
    'uniformly at random'. If that's the case, then the expected
    value of a node's children is the average of their values.
    """
    legalActions = gameState.getLegalActions(agentIndex)   
    successorStates = [gameState.generateSuccessor(agentIndex, action) for action in legalActions]

    if self.gameOver(gameState, depth): # at an end
      return self.evaluationFunction(gameState)
    else:

      #o modulo serve para conseguir voltar ao indice 0
      nextAgent = (agentIndex + 1) % gameState.getNumAgents()
      values = [self.expectimax(state, nextAgent, depth -1) for state in successorStates]            
      
      if agentIndex == 0: # pacman
        return max(values)
      else: # fantasma, onde se aplica o expectmax
        return sum(values)/len(values)

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    depth = gameState.getNumAgents()*self.depth

    legalActions = gameState.getLegalActions(0)
    legalActions.remove(Directions.STOP)
    
    successorStates = [gameState.generateSuccessor(0, action) for action in legalActions]
    # valores para cada estado sucessor
    values = [self.expectimax(state, 1, depth - 1) for state in successorStates]  
             
    # retorna a acao com maior valor associado
    return legalActions[values.index(max(values))]

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
  """
  "*** YOUR CODE HERE ***"
  pos = currentGameState.getPacmanPosition()
  currentScore = scoreEvaluationFunction(currentGameState)

  if currentGameState.isLose(): 
    return -float("inf")
  elif currentGameState.isWin():
    return float("inf")

  # Distancia das comidas
  foodList = currentGameState.getFood().asList()
  manhattanDistanceToClosestFood = min([util.manhattanDistance(pos, food) for food in foodList])
  distanceToClosestFood = manhattanDistanceToClosestFood

  # Comidas restantes
  numberOfFoodsLeft = len(foodList)
  
  # Distancia dos Fantasmas

  # Separacao entre fantasmas ativos e "assustados". Fantasmas sao considerados assustados sempre que pacman esta sobre o efeito da Power Pellet
  scaredGhosts, activeGhosts = [], []
  for ghost in currentGameState.getGhostStates():
    if not ghost.scaredTimer:
      activeGhosts.append(ghost)
    else: 
      scaredGhosts.append(ghost)

  def getManhattanDistances(ghostList): 
    return [util.manhattanDistance(pos, ghost.getPosition()) for ghost in ghostList]

  distanceToClosestActiveGhost = distanceToClosestScaredGhost = 0

  if activeGhosts:
    distanceToClosestActiveGhost = min(getManhattanDistances(activeGhosts))
  else: 
    distanceToClosestActiveGhost = float("inf")
  distanceToClosestActiveGhost = max(distanceToClosestActiveGhost, 5)
    
  if scaredGhosts:
    distanceToClosestScaredGhost = min(getManhattanDistances(scaredGhosts))
  else:
    distanceToClosestScaredGhost = 0 # ignora se o fantasma nao esta assustado
  
  # Power Pellets restantes
  numberOfPowerPelletsLeft = len(currentGameState.getCapsules())
  
  
  #Apos determinar todas as caracteristicas revelantes pra funcao de avaliacao, 
  #multiplicamos cada uma por um coeficiente representando o quanto aquela caracteristica
  #eh "importante" para o comportamento desejado do pacman 

  #Pontuacao atual - Referencia pras demais caracteristicas.
  #Distancia ate a comida mais proxima - Um coeficiente negativo faz com que pacman evite se afastar da comida mais proxima.
  #Quanto menor esse coeficiente, maior a sera a prioridade de pacman em se aproximar as comidas
  
  #Distancia ate o fantasma ativo mais proximo(Mais precisamente, seu inverso) - Um coeficiente negativo ao inverso dessa 
  #caracteristica faz com que pacman evite se aproximar do fantasma ativo mais proximo. Quanto menor esse coeficiente, maior
  #sera a prioridade de pacman em evitar os fantasmas
  
  #Distancia ate o fantasma assustado mais proximo - Um coeficiente negativo faz com que pacman evite se afastar do fantasma
  #assustado mais proximo. Quanto menor esse coeficiente, maior sera a prioridade de pacman em cacar fantasmas assustados
  
  #Numero de Power Pellets restantes - Um coeficiente negativo faz com que pacman evite deixar uma Power Pellet sem ser comida.
  #Quanto menor esse coeficiente, maior sera a prioridade de pacman em comer as Power Pellets se tiver chance.
  
  #Numero de comidas restantes - Um coeficiente negativo faz com que pacman evite deixar uma comida sem ser comida. Quanto menor
  #esse coeficiente, maior sera a prioridade de pacman em comer as comidas se tiver chance

  score = 1    * currentScore + \
          -1.5 * distanceToClosestFood + \
          -2    * (1./distanceToClosestActiveGhost) + \
          -2    * distanceToClosestScaredGhost + \
          -20 * numberOfPowerPelletsLeft + \
          -4    * numberOfFoodsLeft
  return score

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

