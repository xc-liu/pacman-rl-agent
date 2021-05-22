import math
import time

import numpy as np


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


def add_players(idx1, idx2):
    idx1 = int(idx1)
    idx2 = int(idx2)
    bigger_idx = max(idx1, idx2)
    smaller_idx = min(idx1, idx2)
    return int(bigger_idx * 10 + smaller_idx)


def split_players(number):
    number = int(number)
    if (len(str(number))) == 1:
        return False
    if (len(str(number))) == 2:
        idx1 = int(str(number)[0])
        idx2 = int(str(number)[1])
        return idx1, idx2
    if (len(str(number))) == 3:
        idx1 = int(str(number)[0])
        idx2 = int(str(number)[1])
        idx3 = int(str(number)[2])
        return idx1, idx2, idx3
    else:
        idx1 = int(str(number)[0])
        idx2 = int(str(number)[1])
        idx3 = int(str(number)[2])
        idx4 = int(str(number)[3])
        return idx1, idx2, idx3, idx4


def minus_players(number, idx):
    number = int(number)
    idx = int(idx)
    if (len(str(number))) == 1:
        return False
    idx1 = int(number / 10)
    if idx1 != 0:
        idx2 = number - idx1 * 10
    else:
        idx2 = int(number / 10)
        idx1 = number - idx1 * 10
    if idx1 == idx:
        return idx2
    elif idx2 == idx:
        return idx1
    else:
        return False


class stateRepresentation:
    def __init__(self, agent, gameState, index, red):
        self.agent = agent
        self.gameState = gameState
        self.index = index
        self.indices = set(self.agent.getTeam(gameState))
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
        self.corresponding_index = {}
        self.initial_food = 0
        self.digital_state = self.initialise_digital_state(gameState)
        self.next_pos1 = self.initialise_next_pos(steps=1)
        self.next_pos2 = self.initialise_next_pos(steps=2)
        self.score = agent.getScore(gameState)
        self.time_left = gameState.data.timeleft


    def initialise_digital_state(self, gameState):
        # digital state has 7 layers:
        # 0. walls
        # 1. food
        # 2. capsule
        # 3. players
        # 4. directions (ignored for now)
        # 5. pacman players (number = food carried)
        # 6. scared players (number = timer)

        layout = str(gameState.data.layout).split("\n")
        digital_state = np.zeros(shape=(6, len(layout), len(layout[0])))
        for i in range(len(layout)):
            for j in range(len(layout[0])):
                if self.red:
                    idx1 = i
                    idx2 = j
                else:
                    idx1 = len(layout) - i - 1
                    idx2 = len(layout[0]) - j - 1

                if layout[idx1][idx2] == "%":
                    digital_state[0][i][j] = 1
                elif layout[idx1][idx2] == ".":
                    digital_state[1][i][j] = 1
                elif layout[idx1][idx2] == "o":
                    digital_state[2][i][j] = 1
                elif layout[idx1][idx2] != " ":
                    idx = int(layout[idx1][idx2])
                    # our player should always be red team, 1, 3
                    if self.red:
                        digital_state[3][i][j] = idx
                    elif idx in [1, 3]:
                        digital_state[3][i][j] = idx + 1
                    else:
                        digital_state[3][i][j] = idx - 1
                    self.corresponding_index[idx - 1] = digital_state[3][i][j]

                    if digital_state[3][i][j] % 2 == 0:
                        self.last_enemy_pos[int(digital_state[3][i][j])] = (j, len(layout) - i - 1)

        self.initial_food = np.sum(digital_state[1])

        myPos = gameState.getAgentState(self.index).getPosition()
        secondPos = gameState.getAgentState(self.index_second_agent).getPosition()

        if not self.red:
            myPos = (len(layout[0]) - myPos[0] - 1, len(layout) - myPos[1] - 1)
            secondPos = (len(layout[0]) - secondPos[0] - 1, len(layout) - secondPos[1] - 1)

        self.last_team_pos[self.index] = myPos
        self.last_team_pos[self.index_second_agent] = secondPos
        self.initial_team_pos[self.index] = myPos
        self.initial_team_pos[self.index_second_agent] = secondPos

        for i in [2, 4]:
            self.initial_enemy_pos[i] = self.last_enemy_pos[i]
            self.dist_history[i] = [self.agent.distancer.getDistance(pos1=self.last_enemy_pos[i], pos2=myPos)]
            self.dist_history_second_agent[i] = [
                self.agent.distancer.getDistance(pos1=self.last_enemy_pos[i], pos2=secondPos)]

        for i in range(1, 5):
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
        my_prev_food = np.copy(self.digital_state[1, :, :int(width / 2)])
        my_prev_capsule = np.copy(self.digital_state[2, :, :int(width / 2)])
        self.digital_state[1:, :, :] = 0

        # update food
        for i in range(height):
            for j in range(width):
                if self.red:
                    idx1 = j
                    idx2 = height - i - 1
                else:
                    idx1 = width - j - 1
                    idx2 = i

                if r_food[idx1][idx2] or b_food[idx1][idx2]:
                    self.digital_state[1][i][j] = 1

        # if the current food layer is different from the previous one, update the previous position of the enemy
        my_food_change = my_prev_food - self.digital_state[1, :, :int(width / 2)]
        change_food = np.nonzero(my_food_change)
        change_pos = [(b, height - a - 1) for a in change_food[0] for b in change_food[1]]

        # update capsule
        if self.red:
            if len(r_capsule) > 0:
                for cap in r_capsule:
                    self.digital_state[2][height - 1 - cap[1]][cap[0]] = 1
            if len(b_capsule) > 0:
                for cap in b_capsule:
                    self.digital_state[2][height - 1 - cap[1]][cap[0]] = 1
        else:
            if len(r_capsule) > 0:
                for cap in r_capsule:
                    self.digital_state[2][cap[1]][width - 1 - cap[0]] = 1
            if len(b_capsule) > 0:
                for cap in b_capsule:
                    self.digital_state[2][cap[1]][width - 1 - cap[0]] = 1

        my_capsule_change = my_prev_capsule - self.digital_state[2, :, :int(width / 2)]
        change_capsule = np.nonzero(my_capsule_change)
        change_pos += [(b, height - a - 1) for a in change_capsule[0] for b in change_capsule[1]]

        # update player states
        myPos = gameState.getAgentState(self.index).getPosition()
        secondPos = gameState.getAgentState(self.index_second_agent).getPosition()

        if not self.red:
            myPos = (width - myPos[0] - 1, height - myPos[1] - 1)
            secondPos = (width - secondPos[0] - 1, height - secondPos[1] - 1)

        for idx in self.agent.getTeam(gameState) + self.agent.getOpponents(gameState):
            enemy = False
            if idx in self.agent.getOpponents(gameState):
                enemy = True

            i_state = gameState.getAgentState(idx)
            pos = i_state.getPosition()
            pacman = i_state.isPacman
            food_carrying = i_state.numCarrying
            scared_timer = i_state.scaredTimer
            # direction = i_state.configuration.direction

            original_idx = idx
            if self.red:
                idx += 1
            else:
                if idx in [0, 2]:
                    idx += 2

            if enemy:

                if pos is None and self.near_last_time(agent_idx=self.index, enemy_idx=idx, distance=2):
                    # if the enemy was right next to us the previous time step but suddenly disappears
                    # it is eaten and back to initial position
                    my_idx = self.corresponding_index[self.index]
                    if (self.last_player_state[my_idx] == "ghost" and self.last_player_state[idx] == "pacman") or \
                            self.last_player_state[idx] == "scared":
                        self.reinitialise_enemy_position(idx, myPos, secondPos)
                    else:
                        self.last_team_pos[self.index] = self.initial_team_pos[self.index]
                        self.dist_history[idx] = [self.agent.distancer.getDistance(self.last_enemy_pos[idx],
                                                                                   self.initial_team_pos[self.index])]

                elif pos is None and self.near_last_time(agent_idx=self.index_second_agent, enemy_idx=idx, distance=2):
                    second_idx = self.corresponding_index[self.index_second_agent]
                    if (self.last_player_state[second_idx] == "ghost" and self.last_player_state[idx] == "pacman") or \
                            self.last_player_state[idx] == "scared":
                        self.reinitialise_enemy_position(idx, myPos, secondPos)
                    else:
                        self.last_team_pos[self.index_second_agent] = self.initial_team_pos[self.index_second_agent]
                        self.dist_history_second_agent[idx] = [
                            self.agent.distancer.getDistance(self.last_enemy_pos[idx],
                                                             self.initial_team_pos[self.index_second_agent])]

                if pos is not None:
                    if not self.red:
                        pos = [width - 1 - pos[0], height - 1 - pos[1]]
                    self.dist_history[idx].append(self.agent.distancer.getDistance(myPos, pos))
                    # self.dist_history[idx] = [self.agent.distancer.getDistance(myPos, pos)]
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
                        noisy_dist = np.clip(self.gameState.getAgentDistances()[original_idx], a_min=5, a_max=None)
                        pos = self.computeOpponentPosition(idx, pacman, noisy_dist, myPos)
            else:
                if not self.red:
                    pos = [width - 1 - pos[0], height - 1 - pos[1]]



            # if self.digital_state[3][height - 1 - int(pos[1])][int(pos[0])] == 0:
            #     self.digital_state[3][height - 1 - int(pos[1])][int(pos[0])] = idx
            # else:
            #     self.digital_state[3][height - 1 - int(pos[1])][int(pos[0])] += idx + 1
            if self.digital_state[3][height - 1 - int(pos[1])][int(pos[0])] == 0:
                self.digital_state[3][height - 1 - int(pos[1])][int(pos[0])] = idx
            else:
                self.digital_state[3][height - 1 - int(pos[1])][int(pos[0])] = add_players(
                    self.digital_state[3][height - 1 - int(pos[1])][int(pos[0])], idx)



            # digital_state[4][height - int(pos[1])][int(pos[0])] = actions_idx[direction]
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
        return self.agent.distancer.getDistance(self.last_enemy_pos[enemy_idx],
                                                self.last_team_pos[agent_idx]) <= distance

    def reinitialise_enemy_position(self, enemy_idx, myPos, secondPos):
        self.last_enemy_pos[enemy_idx] = self.initial_enemy_pos[enemy_idx]
        self.dist_history[enemy_idx] = [self.agent.distancer.getDistance(self.last_enemy_pos[enemy_idx], myPos)]
        self.dist_history_second_agent[enemy_idx] = [
            self.agent.distancer.getDistance(self.last_enemy_pos[enemy_idx], secondPos)]

    def update_last_enemy_positions(self, agentDistancs, enemy_pacman):
        # this function is used for the second agent to update the last enemy positions
        # when the enemies are too far to detect true positions
        for idx in self.agent.getOpponents(self.gameState):
            original_idx = idx
            if self.red:
                idx += 1
            else:
                if idx in [0, 2]:
                    idx += 2
            noisy_dist = np.clip(agentDistancs[original_idx], a_min=5, a_max=None)
            secondPos = self.gameState.getAgentState(self.index_second_agent).getPosition()
            self.computeOpponentPosition(idx, enemy_pacman[original_idx], noisy_dist, secondPos, "second")

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
        if (self.red and enemy_pacman) or (not self.red and enemy_pacman):
            best_enemy_pos = (int(min(best_enemy_pos[0], len(self.digital_state[0][0]) / 2)), best_enemy_pos[1])
            while self.digital_state[0][best_enemy_pos[1]][len(self.digital_state[0][0]) - 1 - best_enemy_pos[0]] == 1:
                best_enemy_pos = (best_enemy_pos[0] - 1, best_enemy_pos[1])

        if (self.red and not enemy_pacman) or (not self.red and not enemy_pacman):
            best_enemy_pos = (int(max(best_enemy_pos[0], int(len(self.digital_state[0][0]) / 2 + 1))), best_enemy_pos[1])
            while self.digital_state[0][best_enemy_pos[1]][len(self.digital_state[0][0]) - 1 - best_enemy_pos[0]] == 1:
                best_enemy_pos = (best_enemy_pos[0] + 1, best_enemy_pos[1])

        self.last_enemy_pos[enemy_idx] = best_enemy_pos
        return self.last_enemy_pos[enemy_idx]

    def get_state_info(self, reshape=False):
        if reshape:
            digital_state = self.reshape_state(self.digital_state)
        else:
            digital_state = self.digital_state

        return digital_state, self.score, self.time_left

    def get_dense_state_representation(self, agent_idx):
        player_layer = np.zeros(shape=self.digital_state[3].shape)
        player_loc = np.nonzero(self.digital_state[3])
        player_loc = [(player_loc[0][i], player_loc[1][i]) for i in range(len(player_loc[0]))]
        indices = self.indices.copy()
        indices.discard(agent_idx)
        agent_idx2 = indices.pop()
        player_idx_map = {int(self.corresponding_index[agent_idx]): 1, int(self.corresponding_index[agent_idx2]): 2, 2: 4, 4: 4}
        info_idx_map = {int(self.corresponding_index[agent_idx]): 0, int(self.corresponding_index[agent_idx2]): 1, 2: 2, 4: 3}

        food_carrying = [0, 0, 0, 0]
        scared_timer = [0, 0, 0, 0]
        for loc in player_loc:
            player = int(self.digital_state[3][loc[0]][loc[1]])
            if player < 10:
                player_layer[loc[0]][loc[1]] = player_idx_map[player]
                food_carrying[info_idx_map[player]] = self.digital_state[4][loc[0]][loc[1]]
                scared_timer[info_idx_map[player]] = self.digital_state[5][loc[0]][loc[1]]
            else:
                players = split_players(player)
                try:
                    player_layer[loc[0]][loc[1]] = min(sum([player_idx_map[p] for p in players]), 4)
                    for p in players:
                        food_carrying[info_idx_map[p]] = self.digital_state[4][loc[0]][loc[1]] / 2
                        scared_timer[info_idx_map[p]] = self.digital_state[5][loc[0]][loc[1]] / 2
                except:
                    print(player)
                    print(players)
                    print(player_layer)
                    exit()

        dense_state = np.zeros(shape=self.digital_state[0].shape)
        dense_state += -self.digital_state[0]               # wall: -1
        dense_state += player_layer                         # players: 1, 2, 4, 4 (1+2=3, 4+4=4)
        dense_state += 5 * self.digital_state[1]            # food: 5
        dense_state += 6 * self.digital_state[2]            # capsule:
        return list(dense_state.flatten()) + food_carrying + scared_timer + [self.score, self.time_left]

    def reshape_state(self, state):
        # reshape the state into 3 * 32 * 32
        reshaped = np.zeros(shape=(3, 32, 32))
        for i in range(3):
            reshaped[i, :16, :] = state[2 * i, :, :]
            reshaped[i, 16:, :] = state[2 * i + 1, :, :]
        return reshaped

    def visualise_reshaped_state(self, reshaped):
        digital_state = np.zeros(shape=(6, 16, 32))
        for i in range(3):
            digital_state[2 * i, :, :] = reshaped[i, :16, :]
            digital_state[2 * i + 1, :, :] = reshaped[i, 16:, :]
        self.visualise_digital_state(digital_state)

    def visualise_digital_state(self, digital_state):
        st = ""
        food_carrying = {}
        scared = {}
        for i in range(len(digital_state[0])):
            for j in range(len(digital_state[0][0])):

                if digital_state[3][i][j] != 0:
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
                if layer[i][j] == 0:
                    st += "  "
                else:
                    st += str(int(layer[i][j])) + " "
            st += "\n"
        print(st)

    def check_eaten(self, old_sate, new_state, prev_agent_state, agent_state, enemy_idx):
        enemy_state = old_sate.getAgentState(enemy_idx)
        if self.agent.distancer.getDistance(prev_agent_state.getPosition(), agent_state.getPosition()) <= 1:
            if enemy_state.getPosition() is not None and self.agent.distancer.getDistance(
                    prev_agent_state.getPosition(), enemy_state.getPosition()) <= 2:
                new_enemy_state = new_state.getAgentState(enemy_idx)
                if new_enemy_state.getPosition() is None:
                    return True
        return False

    def get_reward(self, new_state, old_state, mode="individual", agent_idx=None):
        if mode == "individual":
            assert agent_idx is not None
            indices = [agent_idx]
        else:
            indices = self.agent.getTeam(old_state)

        reward = 0
        for idx in indices:
            agent_reward = 0

            prev_agent_state = old_state.getAgentState(idx)
            agent_state = new_state.getAgentState(idx)
            if (agent_state.numReturned - prev_agent_state.numReturned) <= 0:
                agent_reward += (agent_state.numCarrying - prev_agent_state.numCarrying) / self.initial_food

            agent_reward += (agent_state.numReturned - prev_agent_state.numReturned) / self.initial_food

            powered = False
            enemy_food = 0
            for enemy_idx in self.agent.getOpponents(old_state):
                enemy_state = new_state.getAgentState(enemy_idx)
                enemy_food += enemy_state.numCarrying
                prev_enemy_state = old_state.getAgentState(enemy_idx)
                if (enemy_state.numReturned - prev_enemy_state.numReturned)<=0:
                    food_carrying_diff = enemy_state.numCarrying - prev_enemy_state.numCarrying
                    agent_reward -= (food_carrying_diff / self.initial_food) / len(indices)
                if enemy_state.scaredTimer > 0 and prev_enemy_state.scaredTimer == 0:
                    powered = True

                if self.check_eaten(old_state, new_state, prev_agent_state, agent_state, enemy_idx):
                    agent_reward += 0.2

                agent_reward -= ((enemy_state.numReturned - prev_enemy_state.numReturned) / self.initial_food) / len(
                    indices)

            if self.agent.distancer.getDistance(prev_agent_state.getPosition(), agent_state.getPosition()) > 1:
                agent_reward -= 0.2

            if powered:
                agent_reward += 0.2 / len(indices)
            if agent_state.scaredTimer > 0 and prev_agent_state.scaredTimer == 0:
                agent_reward -= 0.2 / len(indices)

            reward = agent_reward

        return reward

    def get_positional_reward(self, new_state, old_state, distancer, mode="individual", agent_idx=None,
                              annealing_start=None, current_timestep=None):
        if mode == "individual":
            assert agent_idx is not None
            indices = [agent_idx]
        else:
            indices = self.agent.getTeam(old_state)

        reward = 0
        for idx in indices:
            agent_reward = 0

            prev_agent_state = old_state.getAgentState(idx)
            agent_state = new_state.getAgentState(idx)
            prev_agent_state_pos = prev_agent_state.getPosition()
            agent_state_pos = agent_state.getPosition()

            if self.red:
                food = old_state.getBlueFood()
            else:
                food = old_state.getRedFood()

            min_dist = 1e4
            agent_state_pos = (int(agent_state_pos[0]), int(agent_state_pos[1]))
            prev_agent_state_pos = (int(prev_agent_state_pos[0]), int(prev_agent_state_pos[1]))
            prev_min_dist = 1e4
            for i in range(food.width):
                for j in range(food.height):
                    if food[i][j]:
                        if distancer.getDistance(agent_state_pos, (i, j)) < min_dist:
                            min_dist = distancer.getDistance(agent_state_pos, (i, j))
                        if distancer.getDistance(prev_agent_state_pos, (i, j)) < prev_min_dist:
                            prev_min_dist = distancer.getDistance(prev_agent_state_pos, (i, j))

            if agent_state.numCarrying == 0:
                dist_diff = (prev_min_dist - min_dist)
                go_to_food_reward = dist_diff * (37 * prev_min_dist ** 2 + 1975 * prev_min_dist + 242200) / 12650000
                if annealing_start is None:
                    agent_reward += go_to_food_reward
                else:
                    assert current_timestep is not None and current_timestep >= annealing_start
                    agent_reward += go_to_food_reward * 0.9997 ** (current_timestep - annealing_start)
            if agent_state_pos == prev_agent_state_pos:
                if annealing_start is None:
                    agent_reward -= 1
                else:
                    assert current_timestep is not None and current_timestep >= annealing_start
                    agent_reward -= 0.99977 ** (current_timestep - annealing_start)

            # prev_agent_distances = old_state.getAgentDistances()
            # agent_distances = new_state.getAgentDistances()
            # for enemy_idx in self.agent.getOpponents(old_state):
            #     enemy_state = new_state.getAgentState(enemy_idx)
            #
            #     dist = agent_distances[enemy_idx]
            #     prev_dist = prev_agent_distances[enemy_idx]
            #     if agent_state.isPacman and not enemy_state.isPacman and enemy_state.scaredTimer == 0:
            #         agent_reward += dist - prev_dist
            #     if not agent_state.isPacman and enemy_state.isPacman and agent_state.scaredTimer == 0:
            #         agent_reward += prev_dist - dist
            #     if not agent_state.isPacman and enemy_state.isPacman and agent_state.scaredTimer > 0:
            #         if dist < agent_state.scaredTimer + 2:
            #             agent_reward += dist - prev_dist
            reward += agent_reward
        return reward