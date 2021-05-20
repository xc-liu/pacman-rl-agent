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
from MiniMax import MiniMax
from Mini_max_game import Mini_Max_Game
import os
#################
# Team creation #
#################

is_red = True
mini_max = MiniMax()
game_state = None
first_to_act = None

flip = {'North': 'South', 'East': 'West', 'South': 'North', 'West': 'East'}

def createTeam(firstIndex, secondIndex, isRed,
               first='DummyAgent', second='DummyAgent'):
    global first_index, second_index, is_red
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
    is_red = isRed
    first_index = firstIndex
    second_index = secondIndex
    os.system('cls')
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class DummyAgent(CaptureAgent):
    agent_index = None
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at baselineTeam.py for more details about how to
    create an agent as this is the bare minimum.
    """

    def registerInitialState(self, gameState):
        global mini_max, game_state, first_to_act
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

        if first_to_act is None:
            mini_max.player_idx = self.getTeam(gameState)
            mini_max.enemy_idx = self.getOpponents(gameState)
            mini_max.agent = self
            mini_max.update_players()

            game_state = Mini_Max_Game(self, gameState, self.index, is_red, self.distancer)
            first_to_act = self.index


    def chooseAction(self, gameState):
        global is_red, game_state, first_to_act

        if self.index==first_to_act:
            game_state.update_state(gameState)
        else:
            opponent_pacman = {}
            for i in self.getOpponents(gameState):
                opponent_pacman[i] = gameState.getAgentState(i).isPacman
            game_state.update_last_enemy_positions(gameState.getAgentDistances(), opponent_pacman)

        legal_moves = gameState.getLegalActions(self.index)
        move = mini_max.search_best_move(game_state, self.index, legal_moves)
        # if move not in legal_moves: move = flip[move]
        return move


