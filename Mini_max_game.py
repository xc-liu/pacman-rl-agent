import random
import time

import numpy as np
from copy import deepcopy
from Child_Game import Child_Game

action_tuple_red = {"North": (0, -1),
                "South": (0, 1),
                "East": (-1, 0),
                "West": (1, 0),
                "Stop": (0, 0)}

tuple_action_red = {(0, -1): "North",
                (0, 1): "South",
                (-1, 0): "East",
                (1, 0): "West",
                (0, 0): "Stop"}

action_tuple_blue = {"North": (0, 1),
                "South": (0, -1),
                "East": (1, 0),
                "West": (-1, 0),
                "Stop": (0, 0)}

tuple_action_blue = {(0, 1): "North",
                (0, -1): "South",
                (1, 0): "East",
                (-1, 0): "West",
                (0, 0): "Stop"}


def kalman(v, r, q):
    """
      v - array to filter
      t - time steps, usually just range(len(v))
      r - how much we trust our path def=0.1
      q - decreasing it makes it smoother def=0.01
    """
    p = 12
    x = 0
    for i in range(len(v)):
        p = p + q
        K = p * (1 / (p + r))
        x = x + K * (v[i] - x)
        p = (1 - K) * p
    return x


def find_nearest(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


class Mini_Max_Game:
    def __init__(self, agent, gameState, index, red, distancer):
        self.agent = agent
        self.gameState = gameState
        self.index = index
        indices = set(self.agent.getTeam(gameState))
        indices.discard(self.index)
        self.index_second_agent = indices.pop()
        self.red = red
        self.dist_history = {}
        self.dist_history_second_agent = {}
        self.initial_enemy_pos = {}
        self.initial_team_pos = {}
        self.last_enemy_pos = {}
        self.last_team_pos = {}
        self.last_player_state = {}
        self.digital_state = self.initialise_digital_state(gameState)
        self.next_pos1 = self.initialise_next_pos(steps=1)
        self.next_pos2 = self.initialise_next_pos(steps=2)
        self.score = agent.getScore(gameState)
        self.time_left = gameState.data.timeleft
        self.distancer = distancer
        if red:
            self.action_tuple = action_tuple_red
            self.tuple_action = tuple_action_red
        else:
            self.action_tuple = action_tuple_blue
            self.tuple_action = tuple_action_blue

    def initialise_digital_state(self, gameState):
        # digital state has 6 layers:
        # 0. walls
        # 1. food
        # 2. capsule
        # 3. players
        # 4. pacman players (number = food carried)
        # 5. scared players (number = timer)

        layout = str(gameState.data.layout).split("\n")
        digital_state = np.zeros(shape=(6, len(layout), len(layout[0])))
        digital_state[3] = -1
        myPos = gameState.getAgentState(self.index).getPosition()
        secondPos = gameState.getAgentState(self.index_second_agent).getPosition()
        for i in range(len(layout)):
            for j in range(len(layout[0])):
                if layout[i][j] == "%":
                    digital_state[0][i][j] = 1
                elif layout[i][j] == ".":
                    digital_state[1][i][j] = 1
                elif layout[i][j] == "o":
                    digital_state[2][i][j] = 1
                elif layout[i][j] != " ":
                    idx = int(layout[i][j]) - 1
                    digital_state[3][i][j] = idx

                    if (self.red and digital_state[3][i][j] % 2 == 1) or (
                            not self.red and digital_state[3][i][j] % 2 == 0):
                        self.last_enemy_pos[idx] = (j, len(layout) - i - 1)
                        self.initial_enemy_pos[idx] = self.last_enemy_pos[idx]
                        self.dist_history[idx] = [
                            self.agent.distancer.getDistance(pos1=self.last_enemy_pos[idx], pos2=myPos)]
                        self.dist_history_second_agent[idx] = [
                            self.agent.distancer.getDistance(pos1=self.last_enemy_pos[idx], pos2=secondPos)]

        self.last_team_pos[self.index] = self.initial_team_pos[self.index] = myPos
        self.last_team_pos[self.index_second_agent] = self.initial_team_pos[self.index_second_agent] = secondPos

        for i in range(4):
            self.last_player_state[i] = "ghost"
        return digital_state

    def initialise_next_pos(self, steps=2):
        next_pos = {}
        h = len(self.digital_state[0])
        w = len(self.digital_state[0][0])
        for i in range(h):
            for j in range(w):
                if self.digital_state[0][h - 1 - i][j] == 0:
                    pos = (i, j)
                    next_pos[(pos[1], pos[0])] = []
                    possible_pos = [(i + p, j + q)
                                    for p in range(-steps, steps + 1)
                                    for q in range(-steps, steps + 1)
                                    if 0 <= i + p < h and 0 <= j + q < w]
                    for p in possible_pos:
                        if self.digital_state[0][h - 1 - p[0]][p[1]] == 0:
                            if self.agent.distancer.getDistance((pos[1], pos[0]), (p[1], p[0])) <= steps:
                                next_pos[(pos[1], pos[0])].append((p[1], p[0]))
        return next_pos

    def update_state(self, gameState):
        self.gameState = gameState
        self.score = self.agent.getScore(gameState)
        self.time_left = gameState.data.timeleft

        height = len(self.digital_state[0])
        width = len(self.digital_state[0][0])

        r_food = gameState.getRedFood()
        b_food = gameState.getBlueFood()
        r_capsule = gameState.getRedCapsules()
        b_capsule = gameState.getBlueCapsules()
        my_prev_food = np.copy(self.digital_state[1, :, :])
        my_prev_capsule = np.copy(self.digital_state[2, :, :])
        self.digital_state[1:, :, :] = 0
        self.digital_state[3] = -1

        # update food
        for i in range(height):
            for j in range(width):
                idx1 = j
                idx2 = height - i - 1

                if r_food[idx1][idx2] or b_food[idx1][idx2]:
                    self.digital_state[1][i][j] = 1

        # if the current food layer is different from the previous one, update the previous position of the enemy
        if self.red:
            my_food_change = my_prev_food[:, :int(width / 2)] - self.digital_state[1, :, :int(width / 2)]
        else:
            my_food_change = my_prev_food[:, int(width / 2):] - self.digital_state[1, :, int(width / 2):]
        change_food = np.nonzero(my_food_change)
        if self.red:
            change_pos = [(b, height - a - 1) for a in change_food[0] for b in change_food[1]]
        else:
            change_pos = [(int(width / 2) + b, height - a - 1) for a in change_food[0] for b in change_food[1]]

        # update capsule
        if len(r_capsule) > 0:
            for cap in r_capsule:
                self.digital_state[2][height - 1 - cap[1]][cap[0]] = 1
        if len(b_capsule) > 0:
            for cap in b_capsule:
                self.digital_state[2][height - 1 - cap[1]][cap[0]] = 1

        if self.red:
            my_capsule_change = my_prev_capsule[:, :int(width / 2)] - self.digital_state[2, :, :int(width / 2)]
        else:
            my_capsule_change = my_prev_capsule[:, int(width / 2):] - self.digital_state[2, :, int(width / 2):]
        change_capsule = np.nonzero(my_capsule_change)
        change_pos += [(b, height - a - 1) for a in change_capsule[0] for b in change_capsule[1]]

        # update player states
        myPos = gameState.getAgentState(self.index).getPosition()
        secondPos = gameState.getAgentState(self.index_second_agent).getPosition()

        for idx in self.agent.getTeam(gameState) + self.agent.getOpponents(gameState):
            enemy = False
            if idx in self.agent.getOpponents(gameState):
                enemy = True

            i_state = gameState.getAgentState(idx)
            pos = i_state.getPosition()
            pacman = i_state.isPacman
            food_carrying = i_state.numCarrying
            scared_timer = i_state.scaredTimer

            if enemy:

                if pos is None and self.near_last_time(agent_idx=self.index, enemy_idx=idx, distance=2):
                    # if the enemy was right next to us the previous time step but suddenly disappears
                    # it is eaten and back to initial position
                    if (self.last_player_state[self.index] == "ghost" and self.last_player_state[idx] == "pacman") or \
                            self.last_player_state[idx] == "scared":
                        self.reinitialise_enemy_position(idx, myPos, secondPos)
                    else:
                        self.last_team_pos[self.index] = self.initial_team_pos[self.index]
                        self.dist_history[idx] = [self.agent.distancer.getDistance(self.last_enemy_pos[idx],
                                                                                   self.initial_team_pos[self.index])]

                elif pos is None and self.near_last_time(agent_idx=self.index_second_agent, enemy_idx=idx, distance=2):
                    if (self.last_player_state[self.index_second_agent] == "ghost" and self.last_player_state[
                        idx] == "pacman") or \
                            self.last_player_state[idx] == "scared":
                        self.reinitialise_enemy_position(idx)
                    else:
                        self.last_team_pos[self.index_second_agent] = self.initial_team_pos[self.index_second_agent]
                        self.dist_history_second_agent[idx] = [
                            self.agent.distancer.getDistance(self.last_enemy_pos[idx],
                                                             self.initial_team_pos[self.index_second_agent])]

                if pos is not None:
                    self.dist_history[idx].append(self.agent.distancer.getDistance(myPos, pos))
                    self.last_enemy_pos[idx] = pos

                else:
                    changed = False
                    if len(change_pos) == 1:
                        indices = list(self.last_enemy_pos.keys())
                        distance_to_food = [self.agent.distancer.getDistance(change_pos[0], p)
                                            for p in self.last_enemy_pos.values()]
                        if distance_to_food[indices.index(idx)] == min(distance_to_food):
                            pos = change_pos[0]
                            changed = True
                            # self.dist_history[idx].append(self.agent.distancer.getDistance(myPos, pos))
                            self.dist_history[idx] = [self.agent.distancer.getDistance(myPos, pos)]
                            self.last_enemy_pos[idx] = pos

                    elif len(change_pos) == 2:
                        indices = list(self.last_enemy_pos.keys())
                        cost1 = self.agent.distancer.getDistance(self.last_enemy_pos[indices[0]], change_pos[0]) \
                                + self.agent.distancer.getDistance(self.last_enemy_pos[indices[1]], change_pos[1])
                        cost2 = self.agent.distancer.getDistance(self.last_enemy_pos[indices[0]], change_pos[1]) \
                                + self.agent.distancer.getDistance(self.last_enemy_pos[indices[1]], change_pos[0])
                        if cost1 < cost2:
                            corresponding_pos = {0: 0, 1: 1}
                        else:
                            corresponding_pos = {0: 1, 1: 0}
                        pos = change_pos[corresponding_pos[indices.index(idx)]]
                        changed = True
                        self.dist_history[idx].append(self.agent.distancer.getDistance(myPos, pos))
                        # self.dist_history[idx] = [self.agent.distancer.getDistance(myPos, pos)]
                        self.last_enemy_pos[idx] = pos

                    if not changed:
                        noisy_dist = np.clip(self.gameState.getAgentDistances()[idx], a_min=5, a_max=None)
                        pos = self.computeOpponentPosition(idx, pacman, noisy_dist, myPos)

            if self.digital_state[3][height - 1 - int(pos[1])][int(pos[0])] == -1:
                self.digital_state[3][height - 1 - int(pos[1])][int(pos[0])] = idx
            else:
                self.digital_state[3][height - 1 - int(pos[1])][int(pos[0])] = add_players(
                    self.digital_state[3][height - 1 - int(pos[1])][int(pos[0])], idx)

            self.digital_state[4][height - 1 - int(pos[1])][int(pos[0])] += food_carrying if pacman else 0
            self.digital_state[5][height - 1 - int(pos[1])][int(pos[0])] += scared_timer

            if pacman:
                self.last_player_state[idx] = "pacman"
            elif scared_timer > 0:
                self.last_player_state[idx] = "scared"
            else:
                self.last_player_state[idx] = "ghost"

        self.last_team_pos[self.index] = myPos
        self.last_team_pos[self.index_second_agent] = secondPos

        return self.digital_state

    def near_last_time(self, agent_idx, enemy_idx, distance):
        self.check_positions()
        return self.agent.distancer.getDistance(self.last_enemy_pos[enemy_idx],
                                                self.last_team_pos[agent_idx]) <= distance

    def reinitialise_enemy_position(self, enemy_idx):
        myPos = self.gameState.getAgentState(self.index).getPosition()
        secondPos = self.gameState.getAgentState(self.index_second_agent).getPosition()

        self.last_enemy_pos[enemy_idx] = self.initial_enemy_pos[enemy_idx]
        self.dist_history[enemy_idx] = [self.agent.distancer.getDistance(self.last_enemy_pos[enemy_idx], myPos)]
        self.dist_history_second_agent[enemy_idx] = [
            self.agent.distancer.getDistance(self.last_enemy_pos[enemy_idx], secondPos)]
        return self.initial_enemy_pos[enemy_idx]

    def computeOpponentPosition(self, enemy_idx, enemy_pacman, noisy_dist, agent_pos, agent="first"):
        # use Kalman filter to correct the noisy distance
        if agent == "first":
            self.dist_history[enemy_idx].append(noisy_dist)
            dist_history = self.dist_history[enemy_idx]
        else:
            self.dist_history_second_agent[enemy_idx].append(noisy_dist)
            dist_history = self.dist_history_second_agent[enemy_idx]

        corrected_dist = np.clip(kalman(dist_history, 0.01, 0.01), a_min=5, a_max=None)
        # corrected_dist = noisy_dist

        # sample around the last enemy state, find the one with closest distance to corrected_dist
        if agent == "first":
            possible_enemy_pos = self.next_pos1[
                (int(self.last_enemy_pos[enemy_idx][0]), int(self.last_enemy_pos[enemy_idx][1]))]
        else:
            possible_enemy_pos = self.next_pos1[
                (int(self.last_enemy_pos[enemy_idx][0]), int(self.last_enemy_pos[enemy_idx][1]))]
        possible_distances = []
        for p in possible_enemy_pos:
            possible_distances.append(self.agent.distancer.getDistance(agent_pos, p))
        best_enemy_pos = possible_enemy_pos[find_nearest(possible_distances, corrected_dist)]

        if (self.red and enemy_pacman) or (not self.red and not enemy_pacman):
            best_enemy_pos = (min(best_enemy_pos[0], len(self.digital_state[0][0]) / 2), best_enemy_pos[1])
            while self.digital_state[0][best_enemy_pos[1]][len(self.digital_state[0][0]) - 1 - best_enemy_pos[0]] == 1:
                best_enemy_pos = (best_enemy_pos[0] - 1, best_enemy_pos[1])

        if (self.red and not enemy_pacman) or (not self.red and enemy_pacman):
            best_enemy_pos = (max(best_enemy_pos[0], int(len(self.digital_state[0][0]) / 2 + 1)), best_enemy_pos[1])
            while self.digital_state[0][best_enemy_pos[1]][len(self.digital_state[0][0]) - 1 - best_enemy_pos[0]] == 1:
                best_enemy_pos = (best_enemy_pos[0] + 1, best_enemy_pos[1])

        self.last_enemy_pos[enemy_idx] = best_enemy_pos
        return self.last_enemy_pos[enemy_idx]

    def update_last_enemy_positions(self, agentDistancs, enemy_pacman):
        # this function is used for the second agent to update the last enemy positions
        # when the enemies are too far to detect true positions
        for idx in self.agent.getOpponents(self.gameState):
            noisy_dist = np.clip(agentDistancs[idx], a_min=5, a_max=None)
            secondPos = self.gameState.getAgentState(self.index_second_agent).getPosition()
            self.computeOpponentPosition(idx, enemy_pacman[idx], noisy_dist, secondPos, "second")

    def visualise_digital_state(self, digital_state):
        st = ""
        food_carrying = {}
        scared = {}
        for i in range(len(digital_state[0])):
            for j in range(len(digital_state[0][0])):

                if digital_state[3][i][j] != -1:
                    st += str(int(digital_state[3][i][j]))
                    if digital_state[4][i][j] != 0:
                        food_carrying[int(digital_state[3][i][j])] = digital_state[4][i][j]
                    if digital_state[5][i][j] != 0:
                        scared[int(digital_state[3][i][j])] = digital_state[5][i][j]
                elif digital_state[1][i][j] == 1:
                    st += "."
                elif digital_state[0][i][j] == 1:
                    st += "%"
                elif digital_state[2][i][j] == 1:
                    st += "o"
                else:
                    st += " "
            st += "\n"
        st = st[:-1]

        info = ""
        if bool(food_carrying):
            info += "Food carrying: "
            for k in food_carrying.keys():
                info += "%d - %d " % (k, food_carrying[k])
        if bool(scared):
            info += "Scared timer: "
            for k in scared.keys():
                info += "%d - %d " % (k, scared[k])

        print(st)
        print(info)
        print()

    def visualise_state(self):
        print("Time left %s, score %s. " % (self.time_left, self.score))
        self.visualise_digital_state(self.digital_state)

    def visualise_one_layer(self, layer):
        st = ""
        for i in range(len(layer)):
            for j in range(len(layer[0])):
                st += str(int(layer[i][j])) + " "
            st += "\n"
        print(st)

    def get_state_info(self):
        return self.digital_state, self.score, self.time_left

    def food_drop_positions(self, position, num_carrying):

        def allGood(x, y):
            pig_length = len(self.digital_state[0][0])
            width, height = pig_length, len(self.digital_state[0])
            food = self.digital_state[1]
            walls = self.digital_state[0]
            capsules = self.digital_state[2]

            # bounds check
            if x >= width or y >= height or x <= 0 or y <= 0:
                return False

            if walls[x][y] == 1:
                return False
            if food[x][y] == 1:
                return False

            # if not onRightSide(state, x, y):
            if (y <= pig_length and self.red) or (y > pig_length and not self.red):
                return False

            if capsules[x][y]:
                return False

            # loop through agents
            if (x, y) in self.last_team_pos.values() or (x, y) in self.last_enemy_pos.values():
                return False

            return True

        def genSuccessors(x, y):
            DX = [-1, 0, 1]
            DY = [-1, 0, 1]
            return [(x + dx, y + dy) for dx in DX for dy in DY]

        positionQueue = [position]
        seen = set()
        numToDump = num_carrying
        food_positions = []
        while numToDump > 0:
            if not len(positionQueue):
                raise Exception('Exhausted BFS! uh oh')
            # pop one off, graph check
            popped = positionQueue.pop(0)
            if popped in seen:
                continue
            seen.add(popped)

            x, y = popped[0], popped[1]
            x = int(x)
            y = int(y)
            if (allGood(x, y)):
                food_positions.append((x, y))
                numToDump -= 1

            # generate successors
            positionQueue = positionQueue + genSuccessors(x, y)
        return food_positions

    def check_positions(self):
        pig_length = len(self.digital_state[0][0])
        pig_height = len(self.digital_state[0])
        players_positions = {}
        player_map = self.digital_state[3]
        players = np.nonzero(player_map + 1)
        players = [(players[0][i], players[1][i]) for i in range(len(players[0]))]
        for p in players:
            players_positions[int(player_map[p[0]][p[1]])] = (p[1], pig_height - 1 - p[0])
        for k in self.last_team_pos:
            for key_ in players_positions:
                if str(k) in str(key_):
                    self.last_team_pos[k] = players_positions[key_]
                    break
        for k in self.last_enemy_pos:
            for key_ in players_positions:
                if str(k) in str(key_):
                    self.last_enemy_pos[k] = players_positions[key_]
                    break

    def update_map(self):
        pig_height = len(self.digital_state[0])
        self.digital_state[3] = -1
        for idx in range(4):
            if idx in self.last_team_pos:
                pos = self.last_team_pos[idx]
            else:
                pos = self.last_enemy_pos[idx]
            if self.digital_state[3][pig_height - 1 - int(pos[1])][int(pos[0])] == -1:
                self.digital_state[3][pig_height - 1 - int(pos[1])][int(pos[0])] = idx
            else:
                self.digital_state[3][pig_height - 1 - int(pos[1])][int(pos[0])] = add_players(
                    self.digital_state[3][pig_height - 1 - int(pos[1])][int(pos[0])], idx)

    def get_next_game(self, action, agent):
        self.visualise_digital_state(self.digital_state)
        print("Parent_game ", agent, action)
        friend = True
        if agent not in self.last_team_pos.keys():
            friend = False

        pig_length = len(self.digital_state[0][0])
        pig_height = len(self.digital_state[0])

        self.check_positions()

        players_positions = {}
        player_map = self.digital_state[3]
        players = np.nonzero(player_map + 1)
        players =[(players[0][i], players[1][i]) for i in range(len(players[0]))]
        for p in players:
            players_positions[int(player_map[p[0]][p[1]])] = (p[1], pig_height - 1 - p[0])
        for k in self.last_team_pos:
            for key_ in players_positions:
                if str(k) in str(key_):
                    self.last_team_pos[k] = players_positions[key_]
                    break
        for k in self.last_enemy_pos:
            for key_ in players_positions:
                if str(k) in str(key_):
                    self.last_enemy_pos[k] = players_positions[key_]
                    break

        new_red = self.red if friend else not self.red
        if friend:
            new_first_index = self.index
            new_second_index = self.index_second_agent
            new_dist_history = dict(self.dist_history)
            new_dist_history_second_agent = dict(self.dist_history_second_agent)
            new_enemy_pos = dict(self.initial_enemy_pos)
            new_team_pos = dict(self.initial_team_pos)
            new_last_enemy_pos = dict(self.last_enemy_pos)
            new_last_team_pos = dict(self.last_team_pos)
        else:
            new_first_index = list(self.dist_history.keys())[0]
            new_second_index = list(self.dist_history.keys())[1]
            new_dist_history = {self.index: self.dist_history[new_first_index],
                                self.index_second_agent: self.dist_history[new_second_index]}
            new_dist_history_second_agent = {self.index: self.dist_history_second_agent[new_first_index],
                                             self.index_second_agent: self.dist_history_second_agent[new_second_index]}
            new_enemy_pos = dict(self.initial_team_pos)
            new_team_pos = dict(self.initial_enemy_pos)
            new_last_enemy_pos = dict(self.last_team_pos)
            new_last_team_pos = dict(self.last_enemy_pos)

        # print(new_first_index, new_second_index, new_red,
        #       new_dist_history, new_dist_history_second_agent, new_enemy_pos,
        #       new_team_pos, new_last_enemy_pos, new_last_team_pos, self.score, self.time_left)

        new_game = Child_Game(self.agent, new_first_index, new_second_index, new_red, self.distancer,
                              new_dist_history, new_dist_history_second_agent, new_enemy_pos,
                              new_team_pos, new_last_enemy_pos, new_last_team_pos, self.last_player_state,
                              self.digital_state, self.next_pos1, self.next_pos2, self.score, self.time_left, self.action_tuple, self.tuple_action)

        friend = True
        if agent not in new_game.last_team_pos.keys():
            friend = False

        new_game.time_left -= 1


        previous_pos = ()
        pos_to_update = ()

        previous_pos = new_game.last_team_pos[agent]
        if action is not "Stop":
            new_position = (new_game.last_team_pos[agent][0] + self.action_tuple[action][0],
                                             new_game.last_team_pos[agent][1] + self.action_tuple[action][1])
            new_game.last_team_pos[agent] = new_position
        pos_to_update = new_game.last_team_pos[agent]
        pos_to_update = (int(pos_to_update[0]), int(pos_to_update[1]))
        previous_pos = (int(previous_pos[0]), int(previous_pos[1]))

        (i, j) = pos_to_update
        (i_p, j_p) = previous_pos
        food_eaten = False

        if action is not "Stop":
            # score and food
            existing = new_game.digital_state[1][pig_height - 1 - j, i]
            if existing == 1:
                food_eaten = True
                new_game.digital_state[1][pig_height - 1 - j, i] = 0

        capsule_map = new_game.digital_state[2]
        existing = capsule_map[pig_height - 1 - j, i]
        capsule_eaten = False
        if existing == 1:
            capsule_eaten = True
            new_game.digital_state[2][pig_height - 1 - j, i] = 0

        if capsule_eaten:
            if friend:
                for p in new_game.last_enemy_pos:
                    (n_i, n_j) = p
                    new_game.digital_state[5][pig_height - 1 - n_j, n_i] = 40
            else:
                for p in new_game.last_team_pos:
                    (n_i, n_j) = p
                    new_game.digital_state[5][pig_height - 1 - n_j, n_i] = 40
        else:
            new_game.digital_state[5] -= 1
            new_game.digital_state[5] = np.clip(new_game.digital_state[5], a_max=None, a_min=0)

        if action is not "Stop":
            new_value = minus_players(new_game.digital_state[3][pig_height - 1 - j_p, i_p], agent)
            new_game.digital_state[3][pig_height - 1 - j_p, i_p] = -1 if type(new_value) == bool else new_value
            agent_idx = int(new_game.digital_state[3][pig_height - 1 - j, i])
            if (agent_idx != -1):
                if agent_idx in new_game.last_enemy_pos:
                    scared_timer = new_game.digital_state[5][pig_height - 1 - j, i]
                    if scared_timer > 0:
                        if (j <= pig_length and self.red) or (j > pig_length and not self.red):
                            num_carrying = new_game.digital_state[4][pig_height - 1 - j_p, i_p]
                            if num_carrying > 0:
                                foods = self.food_drop_positions(pos_to_update, num_carrying)
                                for f in foods:
                                    new_game.digital_state[1][pig_height - 1 - f[1], f[0]] = 1
                            (i, j) = new_game.initial_team_pos[agent]
                        else:
                            new_game.reinitialise_enemy_position(agent_idx)
                    else:
                        if (j <= pig_length and self.red) or (j > pig_length and not self.red):
                            (i_e, j_e) = new_game.reinitialise_enemy_position(agent_idx)
                            new_game.digital_state[5][pig_height - 1 - j_e, i_e] = new_game.digital_state[5][
                                pig_height - 1 - j, i]
                            new_game.digital_state[5][pig_height - 1 - j, i] = 0
                        else:
                            (i_e, j_e) = new_game.initial_team_pos[agent]
                            new_game.digital_state[5][pig_height - 1 - j_e, i_e] = new_game.digital_state[5][
                                pig_height - 1 - j, i]
                            new_game.digital_state[5][pig_height - 1 - j, i] = 0
                            (i, j) = i_e, j_e
                    new_game.digital_state[3][pig_height - 1 - j, i] = agent

                else:
                    players_to_add = list(new_game.last_team_pos.keys())
                    new_game.digital_state[3][pig_height - 1 - j, i] = add_players(players_to_add[0],
                                                                                   players_to_add[1])  # add together
            else:
                new_game.digital_state[3][pig_height - 1 - j, i] = agent

            if food_eaten:
                pacman_food_map = new_game.digital_state[4]
                previous_food = pacman_food_map[pig_height - 1 - j_p, i_p]
                new_game.digital_state[4][pig_height - 1 - j_p, i_p] = 0
                new_game.digital_state[4][pig_height - 1 - j, i] = previous_food + 1
            else:
                pacman_food_map = new_game.digital_state[4]
                previous_food = pacman_food_map[pig_height - 1 - j_p, i_p]
                new_game.digital_state[4][pig_height - 1 - j_p, i_p] = 0
                new_game.digital_state[4][pig_height - 1 - j, i] = previous_food

            carrying_food = new_game.digital_state[4][pig_height - 1 - j, i]
            if (carrying_food > 0):
                if (j <= pig_length and self.red) or (j > pig_length and not self.red):
                    new_game.score += carrying_food
                else:
                    new_game.score -= carrying_food
                new_game.digital_state[4][pig_height - 1 - j, i] = 0
        new_game.visualise_digital_state(new_game.digital_state)
        # print(new_game.index, new_game.index_second_agent, new_game.red,
        #       new_game.dist_history, new_game.dist_history_second_agent, new_game.initial_enemy_pos,
        #       new_game.initial_team_pos, new_game.last_enemy_pos, new_game.last_team_pos, new_game.score, new_game.time_left)
        # print()
        # time.sleep(1)
        # exit()
        return new_game

    def get_legal_actions(self, agent):
        # self.check_positions()
        if agent in self.last_team_pos:
            pos = self.last_team_pos[agent]
        else:
            pos = self.last_enemy_pos[agent]
        if pos not in self.next_pos1:
            new_pos = pos
            while new_pos not in self.next_pos1:
                act = self.action_tuple[random.choice(list(self.action_tuple.keys()))]
                new_pos = (pos[0] + act[0], pos[1] + act[1])
            if agent in self.last_team_pos:
                self.last_team_pos[agent] = new_pos
            else:
                self.last_enemy_pos[agent] = new_pos
            pos = new_pos
            self.update_map()
        possible_next_pos = self.next_pos1[pos]
        possible_moves = []
        for new_pos in possible_next_pos:
            a_tuple = (new_pos[0] - pos[0], new_pos[1] - pos[1])
            possible_moves.append(self.tuple_action[a_tuple])
        return possible_moves

    def get_offensive_heuristic(self, agent_idx):
        pig_length = len(self.digital_state[0][0])
        pig_height = len(self.digital_state[0])

        agent = np.where(self.digital_state[3] == agent_idx)
        if len(agent[0]) == 0:
            agent = np.where(self.digital_state[3] == add_players(self.index, self.index_second_agent))
        agent = (agent[0][0], agent[1][0])
        # agent_position = (agent[1], pig_height - 1 - agent[0])
        # agent_position = (pig_height - 1 - agent[1], agent[0])
        if (self.red and agent[0] <= pig_length) or (not self.red and agent[0] > pig_length):
            food = np.zeros(shape=(len(self.digital_state[0]), pig_length))
            if self.red:
                food[:, int(pig_length / 2) + 1:] = self.digital_state[1, :, int(pig_length / 2) + 1:]
            else:
                food[:, :int(pig_length / 2) + 1] = self.digital_state[1, :, :int(pig_length / 2) + 1]

            min_dist = 1e4
            # for i in range(len(food)):
            #     for j in range(len(food[0])):
            #         if food[i][j]:
            #             self.visualise_state()
            #             if self.distancer.getDistance(agent_position, (pig_height -1 - j, i)) < min_dist:
            #                 min_dist = self.distancer.getDistance(agent_position, (pig_height -1 - j, i))
            for i in range(len(food)):
                for j in range(pig_height):
                    if food[i][j]:
                        self.visualise_state()
                        if self.distancer.getDistance(agent, (i, j)) < min_dist:
                            min_dist = self.distancer.getDistance(agent, (i, j))
            score = -min_dist
        else:
            carrying_food = self.digital_state[4][agent[0]][agent[1]]
            food_returned = self.score
            distance_enemy = 1e4
            enemy_ghost_timer = 0

            for i in self.last_enemy_pos.keys():
                enemy = np.where(self.digital_state[3] == i)
                if len(enemy[0]) == 0:
                    enemy = np.where(self.digital_state[3] == add_players(*list(self.last_enemy_pos.keys())))
                enemy = (enemy[0][0], enemy[1][0])
                dist = abs(enemy[0] - agent[0]) + abs(enemy[1] - agent[1])
                if dist < distance_enemy:
                    distance_enemy = dist
                enemy_ghost_timer += self.digital_state[5][enemy[0]][enemy[1]]

            distance_center = abs(agent[0] - pig_length / 2)

            score = carrying_food + food_returned * 5 + distance_enemy - distance_center * carrying_food + enemy_ghost_timer
        return score

    def get_defensive_heuristic(self, agent):
        if agent in self.last_team_pos:
            pos = self.last_team_pos[agent]
        else:
            pos = self.last_enemy_pos[agent]
        (i, j) = pos
        pig_length = len(self.digital_state[0][0])
        enemy_food_returned = self.score
        being_pacman = int(not (j <= pig_length and self.red) or (j > pig_length and not self.red))

        if self.red:
            remaining_food = np.sum(self.digital_state[1, :, :int(pig_length / 2)])
        else:
            remaining_food = np.sum(self.digital_state[1, :, int(pig_length / 2):])
        distance_center = abs(i - pig_length / 2)

        distance_attacking_enemies = 1e4
        our_ghost_timer = 0
        for i in self.last_enemy_pos.keys():
            enemy = np.where(self.digital_state[3] == i)
            if len(enemy[0]) == 0:
                enemy = np.where(self.digital_state[3] == add_players(*list(self.last_enemy_pos.keys())))
            # self.visualise_state()
            enemy = (enemy[0][0], enemy[1][0])
            dist = abs(enemy[0] - pos[0]) + abs(enemy[1] - pos[1])
            if dist < distance_attacking_enemies:
                distance_attacking_enemies = dist

        for i in self.last_team_pos.keys():
            friend = np.where(self.digital_state[3] == i)
            if len(friend[0]) == 0:
                friend = np.where(self.digital_state[3] == add_players(self.index, self.index_second_agent))
            # self.visualise_state()
            friend = (friend[0][0], friend[1][0])
            our_ghost_timer += self.digital_state[5][friend[0]][friend[1]]

        score = -enemy_food_returned - being_pacman * 10 + remaining_food - our_ghost_timer - distance_center - distance_attacking_enemies / 2
        return score
