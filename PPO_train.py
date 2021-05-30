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

import distanceCalculator
from score_keeper import return_score
from PPO import PPO
from captureAgents import CaptureAgent
from copy import deepcopy
#################
# Team creation #
#################
from stateRepresentation import stateRepresentation

first_index = None  # Index of first player
second_index = None  # Index of second player
ppo_network = PPO()
current_state = None
illegal_reward = None  # Reward in case of illegal movement
actions_idx = {'North': 0, 'South': 1, 'East': 2, 'West': 3, 'Stop': 4}
idx_actions = {0: 'North', 1: 'South', 2: 'East', 3: 'West', 4: 'Stop'}
first_to_initialise = None
first_to_act = None
maze_distancer = None

experience = None
past_gameState = None
previous_agent = None
game_length = None

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

    def check_start(self):
        global first_to_initialise, experience, past_gameState, previous_agent, state, maze_distancer, game_length, state
        if experience is not None and past_gameState is not None:
            first_to_initialise = None
            experience = experience[:-1]
            end_reward = return_score()
            end_game_reward = end_reward if self.red else -end_reward

            # time_left_previous_game = past_gameState.data.timeleft / game_length

            # if end_game_reward > 0:
            #     end_game_reward += 1 + time_left_previous_game*10
            # elif end_game_reward < 0:
            #     end_game_reward -= (1 + time_left_previous_game*10)

            ppo_network.last_experience_reward(end_game_reward)
            experience = (*experience, True, end_game_reward, state.get_dense_state_representation(previous_agent))
            ppo_network.store_experience(experience)
            experience = None
            past_gameState = None
            state = None
            maze_distancer = None


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
        global first_to_initialise, maze_distancer, game_length

        CaptureAgent.registerInitialState(self, gameState)
        if maze_distancer is None:
            maze_distancer = distanceCalculator.Distancer(gameState.data.layout)
            maze_distancer.getMazeDistances()
        self.distancer._distances = None
        self.check_start()

        if game_length is None: game_length = gameState.data.timeleft

        if first_to_initialise is None: first_to_initialise = self.index

        if self.index == first_to_initialise:
            global state
            state = stateRepresentation(self, gameState, self.index, self.red)
        if self.index == first_index:
            self.agent_index = 0
        else:
            self.agent_index = 1


    def chooseAction(self, gameState):
        global experience, past_gameState, illegal_reward, maze_distancer, previous_agent, state

        if experience is not None:
            # reward = state.get_reward(gameState, past_gameState, mode='individual', agent_idx=previous_agent)
            # positional_reward = state.get_positional_reward(gameState, past_gameState, maze_distancer, mode='individual', agent_idx=previous_agent)
            # reward += positional_reward
            # if illegal_reward is not None:
            #     reward -= illegal_reward
            state.update_state(gameState)
            reward = 0
            experience = (*experience, reward, state.get_dense_state_representation(previous_agent))
            ppo_network.store_experience(experience)

        if experience is None: state.update_state(gameState)

        legal_actions = gameState.getLegalActions(self.index)
        legal_actions = [actions_idx[a] for a in legal_actions]
        current_state = state.get_dense_state_representation(self.index)
        current_action, illegal_reward = ppo_network.compute_action(current_state, legal_actions)
        experience = (current_state, current_action, False)
        past_gameState = gameState.deepCopy()
        previous_agent = self.index
        # time.sleep(0.01)
        if illegal_reward is not None:
            return "Stop"
        return idx_actions[current_action]