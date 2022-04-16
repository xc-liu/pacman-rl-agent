# myTeam.py
# ---------
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

#Approximate Q-learning structure extended from: 
#https://github.com/abhinavcreed13/ai-capture-the-flag-pacman-contest/blob/main/myTeamApproxQLearningAgent.py

from captureAgents import CaptureAgent
import random, time, util
from game import Directions, Actions
import game
from util import nearestPoint
import json

#################
# Team creation #
#################

NUM_TRAINING = 0
NUM_GAMES = 0
TRAINING = False

def createTeam(firstIndex, secondIndex, isRed,
               first = 'ApproxQLearningOffense', second = 'ApproxQLearningDefense', **args):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.
  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  #NUM_TRAINING = numTraining
  #NUM_GAMES = numGames
  #TRAINING = training_indicator
  return [eval(first)(firstIndex), eval(second)(secondIndex)]


class ApproxQLearningOffense(CaptureAgent):
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    self.epsilon = 0.1
    self.alpha = 0.2
    self.discount = 0.9
    self.walls = list(gameState.getWalls())
    self.int_walls = [[int(j) for j in i ]for i in self.walls]
    #NUM_TRAINING = self.training_eps
    #TRAINING = self.training
    self.episodesSoFar = 0
    #For competition only:
    #"""
    self.training = False
    self.incoming_weights = 0
    self.incoming_weights_2 = 0
    self.training_eps = 0
    #"""
    #print("Training??")
    #print(self.training)
    #print("what episode?")
    #print(self.training_eps)
    #print("Input weights? (Offense)")
    #print(self.incoming_weights)
    #print(self.episodesSoFar)
    if not self.training:
        #OUR PRE-FINAL POLICY
                        
        #our weights after 50 eps with offense feature group           
        self.weights = {'closest-food': -44.63737147431422, 
        'bias': 41.340963459149194, 
        '#-of-ghosts-1-step-away': -42.84603654346167, 
        '#-of-ghosts-2-steps-away': -5.2173191994697525,
        '#-of-scared-ghosts-1-step-away': 3.7892634277691685e-05, 
        '#-of-scared-ghosts-2-steps-away': 0.0034114311040289156, 
        'ghost-danger': -41.61680439008135, 
        'breakfast-served': 3.7892634277691685e-05, 
        'eats-food': 58.06505777360217, 
        '#-of-invaders-1-step-away': 0.0, 
        '#-of-invaders-2-steps-away': 0.0, 
        '#-of-edible-invaders-1-step-away': 0.3974813297413949, 
        '#-of-edible-invaders-2-steps-away': 1.1409298600894369, 
        'dinner-served': 0.3974813297413949, 
        'home-invasion': 0.0, 
        'enemy-eats-food': -5.171676267058274}
    else:
        self.weights = self.incoming_weights
    
    self.start = gameState.getAgentPosition(self.index)
    self.featuresExtractor = FeaturesExtractor(self)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
        Picks among the actions with the highest Q(s,a).
    """
    
    legalActions = gameState.getLegalActions(self.index)
    if len(legalActions) == 0:
      return None

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in legalActions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start, pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    action = None
    #print("are we still training nao?")
    #print(self.training)
    if self.training:
      #print("scooch amooch")
      for action in legalActions:
        self.updateWeights(gameState, action)
    if not util.flipCoin(self.epsilon) or not self.training:
      # exploit
      #print("WE AINT TRAINING, SHALL EXPLOIT AND WITH THEEESE: ")
      #print(self.weights)
      action = self.getPolicy(gameState)
    else:
      # explore
      #print("EXPLORE LIKE A PIONEER")
      action = random.choice(legalActions)
    return action

  def getWeights(self):
    return self.weights

  def getQValue(self, gameState, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    # features vector
    features = self.featuresExtractor.getFeatures(gameState, action)
    Q = 0
    for feature in self.weights.keys():
      Q += self.weights[feature] * features[feature]
    return Q

  def update(self, gameState, action, nextState, reward):
    """
       Should update your weights based on transition
    """
    features = self.featuresExtractor.getFeatures(gameState, action)
    oldValue = self.getQValue(gameState, action)
    futureQValue = self.getValue(nextState)
    difference = (reward + self.discount * futureQValue) - oldValue
    #print("reward: " + str(reward))
    #print("futureQ: " + str(futureQValue))
    #print("oldVal: " + str(oldValue))
    #print(difference)
    # for each feature i
    for feature in self.weights.keys():
      newWeight = self.alpha * difference * features[feature]
      self.weights[feature] += newWeight
    #print("NEWLY UPDATED WEIGHTS OFFENSE")
    #print(self.weights)
    #print("offensive keys!")
    #print(self.weights.keys())
    

  def updateWeights(self, gameState, action):
    nextState = self.getSuccessor(gameState, action)
    reward = self.getReward(gameState, nextState)
    self.update(gameState, action, nextState, reward)

  def getReward(self, gameState, nextState):
    reward = 0
    agentPosition = gameState.getAgentPosition(self.index)
    enemiesPos = [gameState.getAgentPosition(ene) for ene in self.getOpponents(gameState)]
    #print("here art I: ")
    #print(agentPosition)
    #print("there theys at: ")
    #print(enemiesPos)
    # check if I have updated the score
    #features to represent our team
    #pals = [gameState.getAgentState(i) for i in self.agentInstance.getTeam(gameState) if i != self.agentInstance.index]
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    next_enemies = [gameState.getAgentState(i) for i in self.getOpponents(nextState)]
    
    
    #symmetrical features for the opponents    
    ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None and a.scaredTimer <= 0]
    scared_ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None and a.scaredTimer > 0]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None and gameState.getAgentState(self.index).scaredTimer > 0]
    edible_invaders = [a for a in enemies if a.isPacman and a.getPosition() != None and gameState.getAgentState(self.index).scaredTimer <= 0]

    if self.getScore(nextState) > self.getScore(gameState):
      diff = self.getScore(nextState) - self.getScore(gameState)
      reward += diff * 10

    # check if food eaten in nextState
    myFoods = self.getFood(gameState).asList()
    distToFood = min([self.getMazeDistance(agentPosition, food) for food in myFoods])
    # I am 1 step away, will I be able to eat it?
    if distToFood == 1:
      nextFoods = self.getFood(nextState).asList()
      if len(myFoods) - len(nextFoods) == 1:
        reward += 10
        

    # check if I am eaten
    #enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    #ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    if len(ghosts) > 0:
      minDistGhost = min([self.getMazeDistance(agentPosition, g.getPosition()) for g in ghosts])
      if minDistGhost == 1:
        nextPos = nextState.getAgentState(self.index).getPosition()
        if nextPos == self.start:
          # I die in the next state
          reward += -100
          
          
          
    # check if we eat enemy
    if len(scared_ghosts) > 0:
      minDistGhost = min([self.getMazeDistance(agentPosition, g.getPosition()) for g in scared_ghosts])
      if minDistGhost == 1:
        nextPos = nextState.getAgentState(self.index).getPosition()
        if nextPos in [g.getPosition() for g in scared_ghosts] and len(enemies) > len(next_enemies):
          # I kill in the next state
          reward += 50
          
    if len(edible_invaders) > 0:
      minDistGhost = min([self.getMazeDistance(agentPosition, g.getPosition()) for g in edible_invaders])
      if minDistGhost == 1:
        nextPos = nextState.getAgentState(self.index).getPosition()
        if nextPos in [g.getPosition() for g in edible_invaders] and len(enemies) > len(next_enemies):
          # I kill in the next state
          print("I KILL! (a ghost while atk)")
          reward += 100
    
    return reward

  def final(self, state):
    "Called at the end of each game."
    # call the super-class final method
    CaptureAgent.final(self, state)
    #print("weights in the end (offense)")
    #print(self.weights)
    self.episodesSoFar += 1
    self.incoming_weights = self.weights
    """
    if True:#self.training_eps%50 == 0:
        f = open("weights_offense.txt", "a")
        f.write("Episode " + str(self.training_eps) + "\n")
        weight_string = json.dumps(self.weights)
        f.write(weight_string + "\n")
        f.close()
    """
    # did we finish training?

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def computeValueFromQValues(self, gameState):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    allowedActions = gameState.getLegalActions(self.index)
    if len(allowedActions) == 0:
      return 0.0
    bestAction = self.getPolicy(gameState)
    return self.getQValue(gameState, bestAction)

  def computeActionFromQValues(self, gameState):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    legalActions = gameState.getLegalActions(self.index)
    if len(legalActions) == 0:
      return None
    actionVals = {}
    bestQValue = float('-inf')
    for action in legalActions:
      targetQValue = self.getQValue(gameState, action)
      actionVals[action] = targetQValue
      if targetQValue > bestQValue:
        bestQValue = targetQValue
    bestActions = [k for k, v in actionVals.items() if v == bestQValue]
    # random tie-breaking
    return random.choice(bestActions)

  def getPolicy(self, gameState):
    return self.computeActionFromQValues(gameState)

  def getValue(self, gameState):
    return self.computeValueFromQValues(gameState)


class ApproxQLearningDefense(CaptureAgent):
  def registerInitialState(self, gameState):
    self.epsilon = 0.1
    self.alpha = 0.2
    self.discount = 0.9
    self.start = gameState.getAgentPosition(self.index)
    #NUM_TRAINING = self.training_eps
    #TRAINING = self.training
    self.episodesSoFar = 0
    #For competition only:
    #"""
    self.training = False
    self.incoming_weights = 0
    self.incoming_weights_2 = 0
    self.training_eps = 0
    #"""
    self.walls= list(gameState.getWalls())
    self.int_walls = [[int(j) for j in i ]for i in self.walls]
    #print("Training??")
    #print(self.training)
    #print("what episode?")
    #print(self.training_eps)
    #print("Input weights? (Defense)")
    #print(self.incoming_weights_2)
    #print(self.episodesSoFar)
    if not self.training:
        #OUR PREFINAL WEIGHTS
        #our weights after X eps with defensive feature group
        """
        self.weights = {"closest-prey": 0.007146985500778681, 
                        "bias": -0.3338519152700928, 
                        "#-of-invaders-1-step-away": -1.3270266497784708, 
                        "#-of-invaders-2-steps-away": -5.038870308964941, 
                        "#-of-edible-invaders-1-step-away": 0.0, 
                        "#-of-edible-invaders-2-steps-away": 0.0, 
                        "dinner-served": 0.0, 
                        "enemy-eats-food": 0.00042177103868804293}
        """              
        self.weights = {'closest-prey': -49.887630087162954, 
                        'bias': 0.7930806430268794, 
                        'ghost-danger': -0.08500066897191504, 
                        'breakfast-served': -0.000570929128640594, 
                        '#-of-invaders-1-step-away': 0.0, 
                        '#-of-invaders-2-steps-away': 0.0, 
                        '#-of-edible-invaders-1-step-away': 3, 
                        '#-of-edible-invaders-2-steps-away': 0.09273915171127893, 
                        'dinner-served': 0.3974813297413949, 
                        'home-invasion': 0.0, 
                        'enemy-eats-food': -5.06062414437651}

     
    else:
        self.weights = self.incoming_weights_2
    
    self.start = gameState.getAgentPosition(self.index)
    self.featuresExtractor = FeaturesExtractor(self)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
        Picks among the actions with the highest Q(s,a).
    """
    
    legalActions = gameState.getLegalActions(self.index)
    if len(legalActions) == 0:
      return None

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in legalActions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start, pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    action = None
    #print("are we still training nao?")
    #print(self.training)
    if self.training:
      #print("scooch amooch")
      for action in legalActions:
        self.updateWeights(gameState, action)
    if not util.flipCoin(self.epsilon) or not self.training:
      # exploit
      #print("WE AINT TRAINING, SHALL EXPLOIT AND WITH THEEESE: ")
      #print(self.weights)
      action = self.getPolicy(gameState)
    else:
      # explore
      #print("EXPLORE LIKE A PIONEER")
      action = random.choice(legalActions)
    return action

  def getWeights(self):
    return self.weights

  def getQValue(self, gameState, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    # features vector
    features = self.featuresExtractor.getFeatures(gameState, action)
    Q = 0
    for feature in self.weights.keys():
      Q += self.weights[feature] * features[feature]
    return Q

  def update(self, gameState, action, nextState, reward):
    """
       Should update your weights based on transition
    """
    features = self.featuresExtractor.getFeatures(gameState, action)
    oldValue = self.getQValue(gameState, action)
    futureQValue = self.getValue(nextState)
    difference = (reward + self.discount * futureQValue) - oldValue
    #print("reward: " + str(reward))
    #print("futureQ: " + str(futureQValue))
    #print("oldVal: " + str(oldValue))
    #print(difference)
    # for each feature i
    for feature in self.weights.keys():
      newWeight = self.alpha * difference * features[feature]
      self.weights[feature] += newWeight
    #print("NEWLY UPDATED WEIGHTS DEFENSE")
    #print(self.weights)
    #print("defensive keys!")
    #print(self.weights.keys())

  def updateWeights(self, gameState, action):
    nextState = self.getSuccessor(gameState, action)
    reward = self.getReward(gameState, nextState)
    self.update(gameState, action, nextState, reward)

  def getReward(self, gameState, nextState):
    reward = 0
    agentPosition = gameState.getAgentPosition(self.index)
    enemiesPos = [gameState.getAgentPosition(ene) for ene in self.getOpponents(gameState)]
    #print("here art I: ")
    #print(agentPosition)
    #print("there theys at: ")
    #print(enemiesPos)
    # check if I have updated the score
    #features to represent our team
    #pals = [gameState.getAgentState(i) for i in self.agentInstance.getTeam(gameState) if i != self.agentInstance.index]
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    next_enemies = [gameState.getAgentState(i) for i in self.getOpponents(nextState)]
    
    #symmetrical features for the opponents
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None and gameState.getAgentState(self.index).scaredTimer > 0]
    edible_invaders = [a for a in enemies if a.isPacman and a.getPosition() != None and gameState.getAgentState(self.index).scaredTimer <= 0]
    
    
    if self.getScore(nextState) > self.getScore(gameState):
      diff = self.getScore(nextState) - self.getScore(gameState)
      reward += diff * 10

    # check GUARDED food gets eaten in nextState
    myStash = self.getFoodYouAreDefending(gameState).asList()
    if sum([x != None for x in enemiesPos ]) > 0:
        distToFood = min([self.getMazeDistance(enemyPos, food) for food in myStash for enemyPos in enemiesPos if enemyPos != None])
    # I they are 1 step away, will they be able to eat it?
        if distToFood == 1:
          nextStash = self.getFoodYouAreDefending(nextState).asList()
          if len(myStash) - len(nextStash) == 1:
            reward += -10

    # check if I am eaten
    #enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    #ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
          
    if len(invaders) > 0:
      minDistGhost = min([self.getMazeDistance(agentPosition, g.getPosition()) for g in invaders])
      if minDistGhost == 1:
        nextPos = nextState.getAgentState(self.index).getPosition()
        if nextPos == self.start:
          # I die in the next state
          reward += -50
          
    if len(edible_invaders) > 0:
      minDistGhost = min([self.getMazeDistance(agentPosition, g.getPosition()) for g in edible_invaders])
      if minDistGhost == 1:
        nextPos = nextState.getAgentState(self.index).getPosition()
        if nextPos in [g.getPosition() for g in edible_invaders] and len(enemies) > len(next_enemies):
          # I kill in the next state
          #print("I KILL!")
          reward += 100
      
    return reward

  def final(self, state):
    "Called at the end of each game."
    # call the super-class final method
    CaptureAgent.final(self, state)
    #print("weights in the end (defense)")
    #print(self.weights)
    self.episodesSoFar += 1
    self.incoming_weights_2 = self.weights
    """
    if True:#self.training_eps%50 == 0:
        f = open("weights_defense.txt", "a")
        f.write("Episode " + str(self.training_eps) + "\n")
        weight_string = json.dumps(self.weights)
        f.write(weight_string + "\n")
        f.close()
    """
    # did we finish training?

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def computeValueFromQValues(self, gameState):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    allowedActions = gameState.getLegalActions(self.index)
    if len(allowedActions) == 0:
      return 0.0
    bestAction = self.getPolicy(gameState)
    return self.getQValue(gameState, bestAction)

  def computeActionFromQValues(self, gameState):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    legalActions = gameState.getLegalActions(self.index)
    if len(legalActions) == 0:
      return None
    actionVals = {}
    bestQValue = float('-inf')
    for action in legalActions:
      targetQValue = self.getQValue(gameState, action)
      actionVals[action] = targetQValue
      if targetQValue > bestQValue:
        bestQValue = targetQValue
    bestActions = [k for k, v in actionVals.items() if v == bestQValue]
    # random tie-breaking
    return random.choice(bestActions)

  def getPolicy(self, gameState):
    return self.computeActionFromQValues(gameState)

  def getValue(self, gameState):
    return self.computeValueFromQValues(gameState)
    
    
class FeaturesExtractor:

  def __init__(self, agentInstance):
    self.agentInstance = agentInstance
    self.last_raid = None
    self.bloodlust = 0
    self.empowered = 0
    self.memory = 0
    self.frozen = 0
    self.last_pos = self.agentInstance.start
    self.wildplan = None

  def getFeatures(self, gameState, action):
    #FOLLOWING Gnanasekaran, Feliu Faba and An we use:
    # --> ghosts/scared ghosts 1 and 2 steps away
    # --> binary: ghost prescence 1 step away
    # -->  "eating food"
    # --> distance to closest food
    #the code is still based on Sharma's implementation
    
    #we also add:
    # --> stash (food we guard)
    # --> attacking, defending, defending and scared teammates
    # extract the grid of food and wall locations and get the ghost locations
    if self.bloodlust > 0:
        self.bloodlust -= 1
    if self.empowered > 0:
        self.empowered -= 1
    me = gameState.getAgentState(self.agentInstance.index)
    prev_state = self.agentInstance.getPreviousObservation()
    bag = me.numCarrying
    home = self.agentInstance.start
    food = self.agentInstance.getFood(gameState)
    stash = self.agentInstance.getFoodYouAreDefending(gameState)
    score = self.agentInstance.getScore(gameState)
    RT_score = len(stash.asList()) - len(food.asList()) - bag
    #print("REAL TIME SCORE!")
    #print(RT_score)
    #print("USUAL SCORE")
    #print(score)
    if prev_state is not None:
        prev_stash = self.agentInstance.getFoodYouAreDefending(prev_state)
    stash = self.agentInstance.getFoodYouAreDefending(gameState)
    walls = gameState.getWalls()
    caps = self.agentInstance.getCapsulesYouAreDefending(gameState)
    enemy_caps = self.agentInstance.getCapsules(gameState)
    pals = [gameState.getAgentState(i) for i in self.agentInstance.getTeam(gameState) if i != self.agentInstance.index]
    enemies = [gameState.getAgentState(i) for i in self.agentInstance.getOpponents(gameState)]

    #features to represent our team
    attackers = [a.getPosition() for a in pals if a.isPacman and a.getPosition() != None]
    defenders = [a.getPosition() for a in pals if not a.isPacman and a.getPosition() != None and a.scaredTimer <= 0]
    scared_defenders = [a.getPosition() for a in pals if not a.isPacman and a.getPosition() != None and a.scaredTimer > 0]
    
    #symmetrical features for the opponents
    invaders = [a.getPosition() for a in enemies if a.isPacman and a.getPosition() != None and gameState.getAgentState(self.agentInstance.index).scaredTimer > 0]
    edible_invaders = [a.getPosition() for a in enemies if a.isPacman and a.getPosition() != None and gameState.getAgentState(self.agentInstance.index).scaredTimer <= 0]
    
    ghosts = [a.getPosition() for a in enemies if not a.isPacman and a.getPosition() != None and a.scaredTimer <= 0]
    scared_ghosts = [a.getPosition() for a in enemies if not a.isPacman and a.getPosition() != None and a.scaredTimer > 0]
    
    raid = None
    if prev_state is not None:
        int_stash = [[int(j) for j in i ]for i in stash]
        int_prev_stash = [[int(j) for j in i ]for i in prev_stash]
        raid = None
        for i,row in enumerate(int_stash):
            for j, ele in enumerate(row):
                if ele != int_prev_stash[i][j]:
                    raid = (i,j)
        #print("THEYRE KILLING US!")
        #print(raid)
    # ghosts = state.getGhostPositions()

    features = util.Counter()

    features["bias"] = 1.0

    # compute the location of pacman after he takes the action
    agentPosition = gameState.getAgentPosition(self.agentInstance.index)
    palx, paly = pals[0].getPosition()
    x, y = agentPosition
    dx, dy = Actions.directionToVector(action)
    next_x, next_y = int(x + dx), int(y + dy)

    #stop freezing!
    _, d2prev = self.a_star_search(self.agentInstance.int_walls, agentPosition, self.last_pos)
    if   d2prev < 2  and score <= 0:
      self.frozen += 1
      #if(self.agentInstance.index == 0):
       # print("IM FREEEEZING")
    else:
      self.frozen = 0

    if self.frozen >= 30 and self.memory == 0:
      #print("NOW IM FROZEN")
      self.memory = 1
      _, self.wildplan = self.furthestFood((int(x), int(y)), stash, walls)
      """
      if home[0] < walls.width/2:
        if self.last_pos[1] < walls.height/2:
          self.wildplan = (int(self.last_pos[0] - walls.width/4), int(0.9*walls.height))
        else:
          self.wildplan = (int(self.last_pos[0] - walls.width/4), int(0.1*walls.height))
      else:
        if self.last_pos[1] < walls.height/2:
          self.wildplan = (int(self.last_pos[0] + walls.width/4), int(0.9*walls.height))
        else:
          self.wildplan = (int(self.last_pos[0] + walls.width/4), int(0.1*walls.height))
      """

    # count the number of ghosts 1-step away
    one_step_away = [Actions.getLegalNeighbors(g, walls) for g in ghosts]
    features["#-of-ghosts-1-step-away"] = sum([(next_x, next_y) in t for t in one_step_away])
    #ADD ALSO GHOSTS 2 STEPS AWAY:
    two_steps_away = []
    for o in one_step_away:
        [two_steps_away.append(Actions.getLegalNeighbors(g, walls)) for g in o]
    features["#-of-ghosts-2-steps-away"] = sum([(next_x, next_y) in t for t in two_steps_away])
    
    #Do the same for scared ghosts
    #1 STEP AWAY:
    one_step_away = [Actions.getLegalNeighbors(g, walls) for g in scared_ghosts]
    features["#-of-scared-ghosts-1-step-away"] = sum([(next_x, next_y) in t for t in one_step_away])
    #2 STEPS AWAY:
    two_steps_away = []
    for o in one_step_away:
        [two_steps_away.append(Actions.getLegalNeighbors(g, walls)) for g in o]
    features["#-of-scared-ghosts-2-steps-away"] = sum([(next_x, next_y) in t for t in two_steps_away])
    
    #"""
    #count (creepy) invaders 1-step away
    one_step_away = [Actions.getLegalNeighbors(g, walls) for g in invaders]
    features["#-of-invaders-1-step-away"] = sum([(next_x, next_y) in t for t in one_step_away])
    #ADD ALSO INVADERS 2 STEPS AWAY:
    two_steps_away = []
    for o in one_step_away:
        [two_steps_away.append(Actions.getLegalNeighbors(g, walls)) for g in o]
    features["#-of-invaders-2-steps-away"] = sum([(next_x, next_y) in t for t in two_steps_away])
    
    #The same for edible invaders:
    #1 STEP AWAY
    one_step_away = [Actions.getLegalNeighbors(g, walls) for g in edible_invaders]
    features["#-of-edible-invaders-1-step-away"] = sum([(next_x, next_y) in t for t in one_step_away])
    #2 STEPS AWAY:
    two_steps_away = []
    for o in one_step_away:
        [two_steps_away.append(Actions.getLegalNeighbors(g, walls)) for g in o]
    features["#-of-edible-invaders-2-steps-away"] = sum([(next_x, next_y) in t for t in two_steps_away])
    #"""

        
    #BINARY: GHOST/EDIBLE PACMAN 1 STEP AWAY
    features["ghost-danger"] = 1.0*(features["#-of-ghosts-1-step-away"] > 0)
    features["dinner-served"] = 1.0*(features["#-of-edible-invaders-1-step-away"] > 0)

    features["home-invasion"] = 1.0*(features["#-of-invaders-1-step-away"] > 0)
    features["breakfast-served"] = 1.0*(features["#-of-scared-ghosts-1-step-away"] > 0)
    
    
    # if len(ghosts) > 0:
    #   minGhostDistance = min([self.agentInstance.getMazeDistance(agentPosition, g) for g in ghosts])
    #   if minGhostDistance < 3:
    #     features["minGhostDistance"] = minGhostDistance

    # successor = self.agentInstance.getSuccessor(gameState, action)
    # features['successorScore'] = self.agentInstance.getScore(successor)

    # if there is no danger of ghosts then add the food feature
    if not features["#-of-ghosts-1-step-away"] and (x, y) in enemy_caps:
      #print("eats capsule!")
      self.empowered = 40

    if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
      features["eats-food"] = 1.0
    
    dist_home = self.getMazeDistancePro((next_x, next_y), (home[0], int(walls.height/2)), walls)
    dist = None
    dncap = None
    if sum([x != None for x in ghosts]) > 0:
        for cap in enemy_caps:
          for ene in ghosts:
            dcap = self.getMazeDistancePro(ene, cap, walls)
            if dcap is not None and (dncap is None or (dncap is not None and dcap > dncap)):
                dncap = dcap
                dist = self.getMazeDistancePro((next_x, next_y), cap, walls)
        #print("getting best capsule!")
        if dist is None or self.empowered > 0:
          #print("wait no, getting best food!")
          dist = self.BestFood((next_x, next_y), ghosts, food, walls)
    else:
        if me.isPacman:
            #print("wonna get the closest foodie to MIG, NOTpal")
            #dist = self.closestFood((int(palx), int(paly)), food, walls)
            dist = self.closestFood((next_x, next_y), food, walls)

        else:
            if self.bloodlust > 0:
              dist = self.getMazeDistancePro((next_x, next_y), self.last_raid, walls)
            else:   
              #"""
              for cap in enemy_caps:
                dcap = self.getMazeDistancePro((next_x, next_y), cap, walls)
                if dcap is not None and (dist is None or (dist is not None and dcap < dist)):
                    #print("gettin crazy cap with no threats at all")
                    dist = dcap
              df = self.closestFood((next_x, next_y), food, walls)
              #"""
              if dist is None or (dist is not None and dist > df):
                #print("hol up, no crazy caps, just wonna get the closest foodie")
                dist = df
                


        
    #for f in food:
        #print("one of the many foodlies")
        #print(f)
    #d = self.a_star_search(self.agentInstance.int_walls, ())
    
    #a_star_search(self, grid: list, begin_point: list, target_point: list, cost=1)
    #dist_2 = self.furthestFood((next_x, next_y), caps, walls)
    
    dist_2 = None
    #"""
    #print("last attack!")
    #print(self.last_raid)
    #print("attack currently")
    #print(raid)
    if raid is not None:
        self.last_raid = raid
        self.bloodlust = 30
    dncap = None
    if sum([x != None for x in edible_invaders]) > 0:
      dist_2 = min([self.getMazeDistancePro((next_x, next_y), edible, walls) for edible in edible_invaders])
      dist_bait = None
      for cap in caps:
        for prey in edible_invaders:
            dcap = self.getMazeDistancePro(prey, cap, walls)
            if dcap is not None and (dncap is None or (dncap is not None and dcap < dncap)):
                dncap = dcap
                if cap == (x, y):
                    dist_bait = None
                else:
                    dist_bait = self.getMazeDistancePro((next_x, next_y), cap, walls)
      if dist_bait is not None:
        dist_2 = min(dist_2, dist_bait)
    else:
      if self.last_raid is not None:
        if self.bloodlust > 0:
         # if(self.agentInstance.index == 2):
          #  print("im going to the place of the invasion!!!")
          dist_2 = self.getMazeDistancePro((next_x, next_y), self.last_raid, walls)
        else:
          #if(self.agentInstance.index == 2):
           # print("blood lust gone")
          mid = int(walls.width/2)
          if home[0] < mid:
            dist_2 = self.getMazeDistancePro((next_x, next_y), (max(int(walls.width/2) - 2, self.last_raid[0]), self.last_raid[1]), walls)
          else:
            dist_2 = self.getMazeDistancePro((next_x, next_y), (min(int(walls.width/2) + 2, self.last_raid[0]), self.last_raid[1]), walls)
      else: 
          #print("patrolll")      
          for cap in caps:
            dcap = self.getMazeDistancePro((next_x, next_y), cap, walls)
            if dcap is not None and (dist_2 is None or (dist_2 is not None and dcap > dist_2)):
              dist_2 = dcap

            
    #print("to food!")
    #print(dist)
    #print("to caps")
    #print(dist_2)
    #print("distance home!" + str(dist_home))   
    
    distToPrey = dist_2

    if self.bloodlust > 0 and bag < 1 and me.scaredTimer == 0:
      dist_partner = self.getMazeDistancePro(pals[0].getPosition(), self.last_raid, walls)
      dist_to_threat = self.getMazeDistancePro((next_x, next_y), self.last_raid, walls)
      if dist_to_threat < dist_partner - 5:
        #if(self.agentInstance.index == 0):
          #print("override! backup")
        dist = dist_to_threat
    if bag + max(RT_score, score - 3) > 0 and not features["ghost-danger"]:
      if me.isPacman:
        #print("override! bag or danger")
        dist = dist_home
      else:
        #print("are we gonna do the little adapted thing that we did?")
        if sum([x != None for x in edible_invaders]) > 0:
          dist = dist_2
        else:
          _, backup_goal = self.furthestFood((int(palx), int(paly)), stash, walls)
          dist = max(self.getMazeDistancePro((next_x, next_y), backup_goal, walls), 10)

    if self.wildplan is not None:
      #print("WAS FROZEN SO GOTTA GET TO:")
      #print(self.wildplan)
      dwp = self.getMazeDistancePro((next_x, next_y), self.wildplan, walls)
      if dwp < 5:
        self.wildplan = None
        self.memory = 0
      else:
        dist = dwp

        
    if features["breakfast-served"]:
      #print("were gonna eat!!")
      dist = min([self.getMazeDistancePro((next_x, next_y), scared_ene, walls) for scared_ene in scared_ghosts])
    if dist is not None:
        features["closest-food"] = float(dist) / (walls.width * walls.height)
    #"""
    #"""
    if distToPrey is not None:
        features["closest-prey"] = float(distToPrey)/ (walls.width * walls.height)
    # capsules = self.agentInstance.getCapsules(gameState)
    # if len(capsules) > 0:
    #   closestCap = min([self.agentInstance.getMazeDistance(agentPosition, cap) for cap in self.agentInstance.getCapsules(gameState)])
    #   features["closestCapsule"] = closestCap
    myStash = stash.asList()
    distToFood = 1000
    if sum([x != None for x in edible_invaders]) > 0:
        #print("PAY ATTENTION THE TEST IS HERE")
        if len(myStash) > 0:
            distToFood = min([self.agentInstance.getMazeDistance(enemyPos, food) for food in myStash for enemyPos in edible_invaders])
        dp = min([self.getMazeDistancePro((next_x, next_y), enemyPos, walls) for enemyPos in edible_invaders])
        # I they are 1 step away, will they be able to eat it?
        if distToFood == 1 and not features["#-of-edible-invaders-1-step-away"]:
            features["enemy-eats-food"] = 1.0

    
    features.divideAll(10.0)
    self.last_pos = agentPosition
    #print(features)
    return features

  def closestFood(self, pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
      pos_x, pos_y, dist = fringe.pop(0)
      if (pos_x, pos_y) in expanded:
        continue
      expanded.add((pos_x, pos_y))
      # if we find a food at this location then exit
      if food[pos_x][pos_y]:
        return dist
      # otherwise spread out from the location to its neighbours
      nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
      for nbr_x, nbr_y in nbrs:
        fringe.append((nbr_x, nbr_y, dist + 1))
    # no food found
    return None
    
  def furthestFood(self, pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    furthest = None
    while fringe:
      pos_x, pos_y, dist = fringe.pop(0)
      if (pos_x, pos_y) in expanded:
        continue
      expanded.add((pos_x, pos_y))
      # if we find a food at this location then exit
      if food[pos_x][pos_y]:
        furthest = dist
        furthest_pos = (pos_x, pos_y)
      # otherwise spread out from the location to its neighbours
      nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
      for nbr_x, nbr_y in nbrs:
        fringe.append((nbr_x, nbr_y, dist + 1))
    # no food found
    return furthest, furthest_pos
    
  def BestFood(self, pos, enemies, food, walls):
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    best = None
    best_d = None
    while fringe:
      pos_x, pos_y, dist = fringe.pop(0)
      if (pos_x, pos_y) in expanded:
        continue
      expanded.add((pos_x, pos_y))
      # if we find a food at this location then exit
      if food[pos_x][pos_y]:
        _, d = self.a_star_search(self.agentInstance.int_walls, pos, (pos_x, pos_y))
        e = min([self.a_star_search(self.agentInstance.int_walls, (int(e_x), int(e_y)), (pos_x, pos_y))[1] for (e_x, e_y) in enemies])
        #print("the epsilon!")
        #print(e)
        if(e-d) > 0:
            best_d = d
        """
        if best is None or (best is not None and (e - d) > best):
            best = e - d
            best_d = d
        """
      # otherwise spread out from the location to its neighbours
      nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
      for nbr_x, nbr_y in nbrs:
        fringe.append((nbr_x, nbr_y, dist + 1))
    # no food found
    return best_d
    
  def getMazeDistancePro(self, pos, goal, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
      pos_x, pos_y, dist = fringe.pop(0)
      if (pos_x, pos_y) in expanded:
        continue
      expanded.add((pos_x, pos_y))
      # if we find a food at this location then exit
      if goal == (pos_x, pos_y):
        return dist
      # otherwise spread out from the location to its neighbours
      nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
      for nbr_x, nbr_y in nbrs:
        fringe.append((nbr_x, nbr_y, dist + 1))
    # no food found
    return None
    
  def a_star_search(self, grid: list, begin_point: list, target_point: list, cost=1):
        if begin_point == target_point:
            return 'Stop', 1
        try:
            assert ((grid[begin_point[0]][begin_point[1]] != 1) and (grid[target_point[0]][target_point[1]] != 1))
        except AssertionError:
            print('Assertion Error: ' + str(grid[begin_point[0]][begin_point[1]]) + ',' + str(grid[target_point[0]][target_point[1]]) + '\n')
            print('begin point: ' + str(begin_point) + ' ' + 'end point: ' + str(target_point))
            quit()
        except TypeError:
            print('begin point: ' + str(begin_point) + ' ' + 'target point: ' + str(target_point) )
        # the cost map which pushes the path closer to the goal
        heuristic = [[0 for row in range(len(grid[0]))] for col in range(len(grid))]
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                try:
                    heuristic[i][j] = abs(i - target_point[0]) + abs(j - target_point[1])
                except TypeError:
                    print('target pos: ' + str(target_point))
                    quit()
                if grid[i][j] == 1:
                    heuristic[i][j] = 99  # added extra penalty in the heuristic map

        # the actions we can take
        delta = [[-1, 0],  # go up
                [0, -1],  # go left
                [1, 0],  # go down
                [0, 1]]  # go right

        close_matrix = [[0 for col in range(len(grid[0]))] for row in range(len(grid))]  # the referrence grid
        close_matrix[begin_point[0]][begin_point[1]] = 1
        action_matrix = [[0 for col in range(len(grid[0]))] for row in range(len(grid))]  # the action grid

        x = begin_point[0]
        y = begin_point[1]
        g = 0
        f = g + heuristic[begin_point[0]][begin_point[1]]
        cell = [[f, g, x, y]]

        found = False  # flag that is set when search is complete
        resign = False  # flag set if we can't find expand

        while not found and not resign:
            if len(cell) == 0:
                resign = True
                return None, None
            else:
                cell.sort()  # to choose the least costliest action so as to move closer to the goal
                cell.reverse()
                next = cell.pop()
                x = next[2]
                y = next[3]
                g = next[1]
                f = next[0]

                if x == target_point[0] and y == target_point[1]:
                    found = True
                else:
                    # delta have four steps
                    for i in range(len(delta)):  # to try out different valid actions
                        x2 = x + delta[i][0]
                        y2 = y + delta[i][1]
                        if x2 >= 0 and x2 < len(grid) and y2 >= 0 and y2 < len(grid[0]):  # 判断可否通过那个点
                            if close_matrix[x2][y2] == 0 and grid[x2][y2] == 0:
                                g2 = g + cost
                                f2 = g2 + heuristic[x2][y2]
                                cell.append([f2, g2, x2, y2])
                                close_matrix[x2][y2] = 1
                                action_matrix[x2][y2] = i

        invpath = []
        x = target_point[0]
        y = target_point[1]
        invpath.append([x, y])  # we get the reverse path from here
        while x != begin_point[0] or y != begin_point[1]:
            x2 = x - delta[action_matrix[x][y]][0]
            y2 = y - delta[action_matrix[x][y]][1]
            x = x2
            y = y2
            invpath.append([x, y])

        path = []
        for i in range(len(invpath)):
            path.append(invpath[len(invpath) - 1 - i])
   
        try:
            x_dir = path[1][0] - path[0][0]
            y_dir = path[1][1] - path[0][1]
        except IndexError:
            print('catching Exception:')
            print(target_point)
            print(begin_point)
            print(len(path))
            print(path[0])
            quit()

        desird_dir = 'Stop'
        # 数组里面的表示是把地图倒着表示的
        if x_dir == -1 and y_dir == 0:
            # return 'North'
            desird_dir = 'West'
        if x_dir == 0 and y_dir == -1:
            # return 'West'
            desird_dir =  'South'

        if x_dir == 1 and y_dir == 0:
            # return 'South'
            desird_dir =  'East'
        if x_dir == 0 and y_dir == 1:
            # return 'East'
            desird_dir = 'North'


        return desird_dir, len(path)    


    