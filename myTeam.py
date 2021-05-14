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
import time

from PPO import PPO
from captureAgents import CaptureAgent
#################
# Team creation #
#################
from stateRepresentation import stateRepresentation

first_index = None  # Index of first player
second_index = None  # Index of second player
ppo_network = PPO()
current_actions = [-1, -1]  # Actions to be executed
temp_actions = [-1, -1]  # Actions in case of illegal movements
current_state = None
illegal_reward = None  # Reward in case of illegal movement
actions_idx = {'North': 0, 'South': 1, 'East': 2, 'West': 3, 'Stop': 4}
idx_actions = {0: 'North', 1: 'South', 2: 'East', 3: 'West', 4: 'Stop'}
first_to_initialise = None
first_to_act = None


def createTeam(firstIndex, secondIndex, isRed,
               first='DummyAgent', second='DummyAgent'):
    global first_index, second_index
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
    first_index = firstIndex
    second_index = secondIndex
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
        global first_to_initialise

        CaptureAgent.registerInitialState(self, gameState)
        self.distancer._distances = None

        if first_to_initialise == None: first_to_initialise = self.index

        if self.index == first_to_initialise:
            global state
            state = stateRepresentation(self, gameState, self.index, self.red)
            state.visualise_state()
        if self.index == first_index:
            self.agent_index = 0
        else:
            self.agent_index = 1


    def chooseAction(self, gameState):
        global current_actions, temp_actions, illegal_reward, first_to_act, current_state

        if first_to_act is None: first_to_act = self.index  # First agent to act during the whole game

        if self.index == first_to_act:
            state.update_state(gameState)
            state.visualise_state()
            time.sleep(0.5)
            legal_actions_1 = gameState.getLegalActions(first_index)
            legal_actions_2 = gameState.getLegalActions(second_index)
            legal_actions_1 = [actions_idx[a] for a in legal_actions_1]
            legal_actions_2 = [actions_idx[a] for a in legal_actions_2]
            current_state = ppo_network.process_state(gameState)
            current_actions, illegal_reward, illegal_idx = ppo_network.compute_action(current_state, legal_actions_1,
                                                                                      legal_actions_2)
            if illegal_reward is not None:  # There is an illegal movement
                if illegal_idx is not None:  # Just one of them is illegal
                    temp_actions = current_actions
                    temp_actions[illegal_idx] = 4
                else:
                    temp_actions = [4, 4]
        else:
            reward = 0  # TODO - Compute reward
            if illegal_reward is not None:
                reward -= illegal_reward
            ppo_network.store_experience(current_state, current_actions[0], current_actions[1], reward,
                                         gameState.isOver())

        if illegal_reward is not None:  # There has been an illegal movement
            if self.index is not first_to_act: illegal_reward = None  # Last movement to be executed
            return idx_actions[temp_actions[self.agent_index]]
        return idx_actions[current_actions[self.agent_index]]