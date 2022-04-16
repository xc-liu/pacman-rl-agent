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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'GreedyAgent', second = 'GreedyAgent'):
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
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)
    '''
    self.currFood = self.getFood(gameState)
    self.foodLeft = len(self.currFood.asList())
    self.currDef = self.getFoodYouAreDefending(gameState)
    self.walls = gameState.getWalls()
    self.width = self.walls.width
    self.height = self.walls.height
    
    self.deadend = self.walls.copy()
    #print(self.deadend)
    
    
    pendinglist = []
    for i in range(self.width):
        for j in range(self.height):
            if (self.deadend[i][j] == False):
                pendinglist.append([i,j])
    #print(pendinglist)
    while len(pendinglist) > 0:
        #print(len(pendinglist))
        x_check, y_check = pendinglist.pop()
        check_list = [[x_check-1, y_check], [x_check+1, y_check], [x_check, y_check-1], [x_check, y_check+1]]
        counter = 0
        for item in check_list:
            if (self.deadend[item[0]][item[1]] == True):
                counter += 1
        if counter >= 3:
            self.deadend[x_check][y_check] = True
            for item in check_list:
                if (self.deadend[item[0]][item[1]] == False):
                    pendinglist.append(item)
    #print(self.deadend)
    '''
        
    
    
    
    

    '''
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    return random.choice(actions)


class GreedyAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
 
  def registerInitialState(self, gameState):
    #self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    
    self.midWidth = gameState.data.layout.width/2
    self.midHeight = gameState.data.layout.height/2

    self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
    #print(self.legalPositions)

    self.distancer.getMazeDistances()

    self.team = self.getTeam(gameState)

    self.enemies = self.getOpponents(gameState)
    
    
    #state fetchfood, retreat, defense
    self.fetchfood = True
    self.safemove = False
    self.retreat = False
    self.defense = False
    
  def getfoodcapcity(self, gameState):
    if (self.getScore(gameState) < 7):
        return 30
    else:
        return 30

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))


    
    foodList = self.getFood(gameState).asList()
    #print(foodList)
    foodLeft = len(foodList)
    myPos = gameState.getAgentPosition(self.index)
    
    #calculate enemy positions
    enemy_dist_dict = dict()
    ignore_distance = 6 # ignore enemies that are too far away
    for enemy in self.enemies:
        enemyPos = gameState.getAgentPosition(enemy)
        if enemyPos:
            enemy_dist_dict[enemy] = min(self.distancer.getDistance(myPos, enemyPos), ignore_distance)
        else:
            enemy_dist_dict[enemy] = ignore_distance
            
    #print(enemy_dist_dict)
    
    if(foodLeft <= 2): #go back and end
        carrynum = gameState.getAgentState(self.index).numCarrying
        if (gameState.getAgentState(self.index).isPacman == True):
            self.SetMode("retreat")
        else:
            self.SetMode("defense")
    
    else:
        #if ghost, defense or fetchfood
        if(gameState.getAgentState(self.index).isPacman == False):
            #if an observable nearby enemy pacman nearby, defense
            #if no observable nearby enemy pacman nearby, fetchfood
            defenseflag = False
            for keys, values in enemy_dist_dict.items():
                if(values <= 4):
                    if(gameState.getAgentState(keys).isPacman == True):
                        if(gameState.getAgentState(self.index).scaredTimer < 2):#agent not scared, consider as a ghost
                            defenseflag = True
            if(defenseflag):
                self.SetMode("defense")
            else:
                self.SetMode("fetchfood")
        
        #if Pacman, fetchfood, safemove or retreat
        else:
            #if no observable nearby enemy ghost nearby and pacman not full, fetchfood
            #if pacman less than half full(<2) and enemy ghost in 3-5, safemove
            #if full or pacman half full(>=2) and a ghost detected in 3-5, or a ghost detected in 1-2, retreat
            
            carrynum = gameState.getAgentState(self.index).numCarrying
            
            if(carrynum >= self.getfoodcapcity(gameState)):#Pacman is full
                self.SetMode("retreat")
            else:
                nearby_ghost_detector = False
                far_ghost_detector = False
                for keys, values in enemy_dist_dict.items():
                    if(values <= 5 and values >= 3):
                        if(gameState.getAgentState(keys).isPacman == False):
                            if(gameState.getAgentState(keys).scaredTimer < 5):#Not scared or scared time is too short, consider it as a ghost
                                far_ghost_detector = True
                    if(values < 3):
                        if(gameState.getAgentState(keys).isPacman == False):
                            if(gameState.getAgentState(keys).scaredTimer < 5):#Not scared or scared time is too short, consider it as a ghost
                                nearby_ghost_detector = True
                                
                if(nearby_ghost_detector == True):#a ghost nearby
                    self.SetMode("retreat")
                elif(far_ghost_detector == True and nearby_ghost_detector == False):#a ghost far away
                    if(carrynum >= self.getfoodcapcity(gameState)/2):#Pacman is half-full
                        self.SetMode("retreat")
                    else:#Pacman not half-full
                        self.SetMode("safemove")
                else:#No ghost detected
                    self.SetMode("fetchfood")
   
    
    if(self.fetchfood):
        #Go to the target food
        
        #Calculate the nearest food
        smallest_distance = self.distancer.getDistance(myPos, foodList[0])
        target = foodList[0]
        
        for foodPos in foodList:
            cur_distance = self.distancer.getDistance(myPos, foodPos)
            if(smallest_distance > cur_distance):
                smallest_distance = cur_distance
                target = foodPos
                
        bestActions = [Directions.STOP]
        bestEvalScore = 0
        
        for action in actions:
            Next_pos = self.getNextPos(myPos, action)
            new_distance = self.distancer.getDistance(Next_pos, target)
            curEvalScore = 1/(new_distance+1)
            
            if(curEvalScore > bestEvalScore):
                bestEvalScore = curEvalScore
                bestActions = [action]
            elif(curEvalScore == bestEvalScore):
                bestActions.append(action)
        
        return random.choice(bestActions)
   
    if(self.safemove):
        #Go fetch food while stay away from enemy ghosts
        
        #Calculate the nearest food
        smallest_distance = self.distancer.getDistance(myPos, foodList[0])
        target = foodList[0]
        
        for foodPos in foodList:
            cur_distance = self.distancer.getDistance(myPos, foodPos)
            if(smallest_distance > cur_distance):
                smallest_distance = cur_distance
                target = foodPos
        
        bestActions = [Directions.STOP]
        bestEvalScore = -float('inf')
        
        for action in actions:
            Next_pos = self.getNextPos(myPos, action)
            new_distance = self.distancer.getDistance(Next_pos, target)
            foodscore = 1/(new_distance+1)
            
            #calculate enemy positions again
            enemy_dist_dict = dict()
            ignore_distance = 6 # ignore enemies that are too far away
            for enemy in self.enemies:
                enemyPos = gameState.getAgentPosition(enemy)
                if enemyPos:
                    enemy_dist_dict[enemy] = min(self.distancer.getDistance(Next_pos, enemyPos), ignore_distance)
                else:
                    enemy_dist_dict[enemy] = ignore_distance
            
            enemy_loss_dict = {0:-1000,1:-10,2:-2,3:-0.2,4:-0.1,5:-0.05,6:0}
            enemy_loss = max([enemy_loss_dict[l] for l in list(enemy_dist_dict.values())])
            
            curEvalScore = foodscore + enemy_loss
            
            if(curEvalScore > bestEvalScore):
                bestEvalScore = curEvalScore
                bestActions = [action]
            elif(curEvalScore == bestEvalScore):
                bestActions.append(action)
                
        return random.choice(bestActions)
    
    if(self.retreat):
        #Go to our side while stay away from enemy ghosts
        
        target = (0,0)
        least_distance = float('inf')
        if(self.red):#red team, myside <= 15
            myside = [14,15]
                        
        else:#blue team, myside >= 16
            myside = [17,18]
        
        for i in myside:
            for j in range(1,17):
                cur_target_pos = (i,j)
                if (cur_target_pos in self.legalPositions):
                    cur_distance = self.distancer.getDistance(myPos, cur_target_pos)
                    if(cur_distance < least_distance):
                        target = cur_target_pos
                        least_distance = cur_distance
            
        
        #target get
        bestActions = [Directions.STOP]
        bestEvalScore = -float('inf')
        
        for action in actions:
            Next_pos = self.getNextPos(myPos, action)
            new_distance = self.distancer.getDistance(Next_pos, target)
            retreatscore = 1/(new_distance+1)
            
            #calculate enemy positions again
            enemy_dist_dict = dict()
            ignore_distance = 6 # ignore enemies that are too far away
            for enemy in self.enemies:
                enemyPos = gameState.getAgentPosition(enemy)
                if enemyPos:
                    enemy_dist_dict[enemy] = min(self.distancer.getDistance(Next_pos, enemyPos), ignore_distance)
                else:
                    enemy_dist_dict[enemy] = ignore_distance
            
            enemy_loss_dict = {0:-1000,1:-10,2:-2,3:-0.2,4:-0.1,5:-0.05,6:0}
            enemy_loss = max([enemy_loss_dict[l] for l in list(enemy_dist_dict.values())])
            
            curEvalScore = retreatscore + enemy_loss
            
            if(curEvalScore > bestEvalScore):
                bestEvalScore = curEvalScore
                bestActions = [action]
            elif(curEvalScore == bestEvalScore):
                bestActions.append(action)
                
        return random.choice(bestActions)
        
        
    
    if(self.defense):
        #Chase nearby enemy pacman
        
        bestActions = [Directions.STOP]
        bestEvalScore = -float('inf')
        
        for action in actions:
            Next_pos = self.getNextPos(myPos, action)
            
            
            #calculate enemy positions again
            enemy_dist_dict = dict()
            ignore_distance = 6 # ignore enemies that are too far away
            for enemy in self.enemies:
                enemyPos = gameState.getAgentPosition(enemy)
                if enemyPos:
                    enemy_dist_dict[enemy] = min(self.distancer.getDistance(Next_pos, enemyPos), ignore_distance)
                else:
                    enemy_dist_dict[enemy] = ignore_distance
            
            defense_dict = {0:1,1:0.5,2:0.2,3:0.05,4:0.01,5:0,6:0}
            defenseScore = sum([defense_dict[l] for l in list(enemy_dist_dict.values())])
            
            curEvalScore = defenseScore
            
            if(curEvalScore > bestEvalScore):
                bestEvalScore = curEvalScore
                bestActions = [action]
            elif(curEvalScore == bestEvalScore):
                bestActions.append(action)
        
        return random.choice(bestActions)
        
        
            
        '''
        for enemy in self.enemies:
            enemyPos = gameState.getAgentPosition(enemy)
            if enemyPos:
                print(enemyPos)
        '''
    

    return random.choice(actions)
    
  def SetMode(self, mode):
    if(mode == "safemove"):
        self.fetchfood = False
        self.safemove = True
        self.retreat = False
        self.defense = False
    elif(mode == "retreat"):
        self.fetchfood = False
        self.safemove = False
        self.retreat = True
        self.defense = False
    elif(mode == "defense"):
        self.fetchfood = False
        self.safemove = False
        self.retreat = False
        self.defense = True
    else:#default as fetchfood
        self.fetchfood = True
        self.safemove = False
        self.retreat = False
        self.defense = False
       

  def getNextPos(self, curpos, action):
    x_axis = curpos[0]
    y_axis = curpos[1]
    '''
    Directions.NORTH: (0, 1),
    Directions.SOUTH: (0, -1),
    Directions.EAST:  (1, 0),
    Directions.WEST:  (-1, 0),
    Directions.STOP:  (0, 0)}
    ''' 
    if action == Directions.NORTH:
        return (x_axis, y_axis + 1)
    if action == Directions.SOUTH:
        return (x_axis, y_axis - 1)
    if action == Directions.EAST:
        return (x_axis + 1, y_axis)
    if action == Directions.WEST:
        return (x_axis - 1, y_axis)
    return (x_axis, y_axis)

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}


