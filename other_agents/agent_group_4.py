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
from baselineTeam import DefensiveReflexAgent, OffensiveReflexAgent, ReflexCaptureAgent
from captureAgents import CaptureAgent
import random, time, util
from random import randint
from game import Directions
import game
import math
from game import Configuration
import copy


#################
# Team creation #
#################
def createTeam(
        firstIndex, secondIndex, isRed
):
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
    agent1 = OffensiveAgent(firstIndex)
    agent2 = DefensiveAgent(secondIndex)
    agent1.teamMate = agent2
    agent2.teamMate = agent1
    return [agent1, agent2]


class OffensiveAgent(OffensiveReflexAgent):

    def __init__(self, index):
        super().__init__(index)
        self.teamMate = None
        self.time_since_last_meal = 0
        self.last_food = 0
        self.prevFoodPos = None
        self.features = None
        self.counter = 0
        self.foodGoal = None

    def getFeatures(self, gameState, action):
        features = util.Counter()
        enemies = approxEnemyPos(self, gameState)
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        foodList = self.getFood(successor).asList()
        # See if we should change food to aim for
        if self.counter % 80 == 0:
            self.foodGoal = random.choice(foodList)
        self.counter += 1

        features["score"] = self.getScore(successor)
        features["foodInStomach"] = myState.numCarrying
        features["foodDist"] = self.getMazeDistance(myPos, self.foodGoal)
        features["homeDist"] = self.getMazeDistance(self.start, myPos)
        features["numcapsules"] = len(self.getCapsules(successor))
        features["distanceToClosestNonScaredEnemy"] = min(
            [self.getMazeDistance(myPos, e.getPosition()) for e in enemies if
             e.getPosition() != None and not e.isPacman and e.scaredTimer == 0], default=11)
        if features["distanceToClosestNonScaredEnemy"] > 10:
            features["distanceToClosestNonScaredEnemy"] = 10
        features['goHomeBonus'] = 1000
        self.features = features
        return features


    def getWeights(self, gameState, action):
        ret = {
            'score': 1000,
            'foodInStomach': 100,
            'distanceToClosestNonScaredEnemy':10
        }
        if self.features['foodInStomach'] > 0:
            if self.features['distanceToClosestNonScaredEnemy'] > 7 and self.features["foodInStomach"] < 7:
                # Have food in stomach, but are not close to enemy
                ret['foodDist'] = -5
            else:
                # Enemy close or a lot of food in stomach, go home
                ret['homeDist'] = -5
                ret['goHomeBonus'] = 1
                ret['numCapsules'] = -1
        else:
            ret['foodDist'] = -5
        return ret


class DefensiveAgent(DefensiveReflexAgent):

    def __init__(self, index):
        super().__init__(index)
        self.teamMate = None
        self.prevFoodPos = None
        self.eatenFoodPos = None

    def getFeatures(self, gameState, action):
        features = util.Counter()
        self.eatenFoodPos = checkEatenFood(self, gameState)
        enemies = approxEnemyPos(self, gameState, self.eatenFoodPos)
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman:
            features['onDefense'] = 0
        # Computes distance to invaders we can see
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists) if myState.scaredTimer == 0 else -min(dists)

        features["food"] = len(self.getFoodYouAreDefending(successor).asList())
        features["capsules"] = min(
            [self.getMazeDistance(myPos, capsule) for capsule in self.getCapsulesYouAreDefending(successor)], default=0)

        features["averagedistancetofood"] = sum([self.getMazeDistance(myPos, f) for f in self.getFoodYouAreDefending(successor).asList()])/len(self.getFoodYouAreDefending(successor).asList())

        return features

    def getWeights(self, gameState, action):
        return {
            'numInvaders': -1000,
            'onDefense': 100,
            'invaderDistance': -10,
            "food": 1,
            "capsules": -5,
        }

def approxEnemyPos(agent, gamestate, eatenFoodPos = None):
    enemies = [(i, copy.deepcopy(gamestate.getAgentState(i))) for i in agent.getOpponents(gamestate)]  # [(index, enemyState), ....]
    for (enemyInd, enemy)  in enemies:
        if enemy.getPosition() != None: continue

        teamMate = agent.teamMate
        if agent.getPreviousObservation() is None or teamMate.getPreviousObservation() is None:
            # First iteration, return start positions
            enemy.configuration = enemy.start
            continue

        if eatenFoodPos != None:
            enemy.configuration = Configuration(eatenFoodPos, direction='Stop')
            continue

        agentCurrDists = gamestate.getAgentDistances()[enemyInd]    # [31 ... 43]
        agentPrevDists = agent.getPreviousObservation().getAgentDistances()[enemyInd]  #  # [28 ... 42]
        teamMateCurrDists = teamMate.getCurrentObservation().getAgentDistances()[enemyInd]
        teamMatePrevDists = teamMate.getPreviousObservation().getAgentDistances()[enemyInd]

        manhattan_agent_min = max(agentCurrDists-6, agentPrevDists-8)
        manhattan_agent_max = min(agentCurrDists + 6, agentPrevDists + 8)

        manhattan_teamMate_min = max(teamMateCurrDists - 6, teamMatePrevDists - 8)
        manhattan_teamMate_max = min(teamMateCurrDists + 6, teamMatePrevDists + 8)

        agent_pos = gamestate.getAgentPosition(agent.index)
        teamMate_pos = gamestate.getAgentPosition(teamMate.index)

        # possible_pos = [[False for _ in range(16)] for _ in range(32)]
        possible_pos = []
        for i in range(26):
            for j in range(14):
                if gamestate.hasWall(i, j): continue
                manhattan_agent_dist = abs(agent_pos[0] - i) + abs(agent_pos[1] - j)
                manhattan_teamMate_dist = abs(teamMate_pos[0] - i) + abs(teamMate_pos[1] - j)
                if (manhattan_agent_min <= manhattan_agent_dist <= manhattan_agent_max) and (manhattan_teamMate_min <= manhattan_teamMate_dist <= manhattan_teamMate_max):
                    possible_pos.append((i, j))
        #agent.debugDraw(possible_pos, [1, 0, 0], clear=True)
        if len(possible_pos) == 0:
            enemy.configuration = enemy.start
        else:
            enemy.configuration = Configuration(random.choice(possible_pos), direction='Stop')

    # enemy1Pos = enemies[0][1].configuration.pos
    # enemy2Pos = enemies[1][1].configuration.pos
    # agent.debugDraw(enemy1Pos, [1, 0, 0], clear=True)
    # agent.debugDraw(enemy2Pos, [1, 0, 0], clear=True)
    return [enemy for _, enemy in enemies]

def checkEatenFood(agent, gameState):
    foodPositions = agent.getFoodYouAreDefending(gameState).asList()
    eatenFoodPos = agent.eatenFoodPos
    if agent.prevFoodPos != None:
        if len(foodPositions) != len(agent.prevFoodPos):  # SOMEONE HAS EATEN OUR FOOD!!
            # Get food that has been eaten
            for i in range(min(len(foodPositions), len(agent.prevFoodPos))):
                if agent.prevFoodPos[i] != foodPositions[i]:
                    # We have found the position of the eaten food. Find and accuse closest enemy
                    eatenFoodPos = agent.prevFoodPos[i]
                    break

    agent.prevFoodPos = foodPositions
    return eatenFoodPos