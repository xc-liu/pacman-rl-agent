import time
import random

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
    idx1 = int(number / 10)
    idx2 = number - idx1 * 10
    return idx1, idx2


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


class Child_Game:
    def __init__(self, agent, index, index_second_agent, red, distancer, dist_history, dist_history_second_agent,
                 initial_enemy_pos, initial_team_pos, last_enemy_pos, last_team_pos, last_player_state, digital_state,
                 next_pos1, next_pos2, score, time_left, action_tuple, tuple_action):
        self.agent = agent
        self.index = index
        self.index_second_agent = index_second_agent
        self.red = red
        self.dist_history = dist_history
        self.dist_history_second_agent = dist_history_second_agent
        self.initial_enemy_pos = initial_enemy_pos
        self.initial_team_pos = initial_team_pos
        self.last_enemy_pos = last_enemy_pos
        self.last_team_pos = last_team_pos
        self.last_player_state = last_player_state
        self.digital_state = digital_state
        self.next_pos1 = next_pos1
        self.next_pos2 = next_pos2
        self.score = score
        self.time_left = time_left
        self.distancer = distancer
        self.action_tuple = action_tuple
        self.tuple_action = tuple_action

    def reinitialise_enemy_position(self, enemy_idx):
        pig_height = len(self.digital_state[0])

        agent1 = np.where(self.digital_state[3] == self.index)
        agent2 = np.where(self.digital_state[3] == self.index_second_agent)
        if len(agent1[0]) == 0:
            agent1 = agent2 = np.where(self.digital_state[3] == add_players(self.index, self.index_second_agent))
        self.visualise_state()
        print(self.last_team_pos)
        # print(self.digital_state[3])
        print(self.index)
        print(self.index_second_agent)
        print(agent1)
        print(agent2)
        agent1 = (agent1[1][0], pig_height- 1 - agent1[0][0])
        agent2 = (agent2[1][0], pig_height- 1 - agent2[0][0])

        self.last_enemy_pos[enemy_idx] = self.initial_enemy_pos[enemy_idx]
        self.dist_history[enemy_idx] = [self.agent.distancer.getDistance(self.last_enemy_pos[enemy_idx], agent1)]
        self.dist_history_second_agent[enemy_idx] = [
            self.agent.distancer.getDistance(self.last_enemy_pos[enemy_idx], agent2)]
        return self.initial_enemy_pos[enemy_idx]

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
        # self.visualise_digital_state(self.digital_state)
        # print("Chield_game ", agent, action)
        friend = True
        if agent not in self.last_team_pos.keys():
            friend = False

        pig_length = len(self.digital_state[0][0])
        pig_height = len(self.digital_state[0])

        self.check_positions()

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
                              self.digital_state, self.next_pos1, self.next_pos2, self.score, self.time_left,
                              self.action_tuple, self.tuple_action)

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
        # new_game.visualise_digital_state(new_game.digital_state)
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

    def get_offensive_heuristic(self, agent_idx):
        pig_length = len(self.digital_state[0][0])
        pig_height = len(self.digital_state[0])

        second_idx = agent_idx + 2 if agent_idx < 2 else agent_idx-2

        agent = np.where(self.digital_state[3] == agent_idx)
        if len(agent[0]) == 0:
            agent = np.where(self.digital_state[3] == add_players(agent_idx, second_idx))
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
        for e in self.last_enemy_pos.keys():
            enemy = np.where(self.digital_state[3] == e)
            if len(enemy[0]) == 0:
                enemy = np.where(self.digital_state[3] == add_players(*list(self.last_enemy_pos.keys())))
            enemy = (enemy[0][0], enemy[1][0])
            dist = abs(enemy[0] - pos[0]) + abs(enemy[1] - pos[1])
            if dist < distance_attacking_enemies:
                distance_attacking_enemies = dist

        for e in self.last_team_pos.keys():
            friend = np.where(self.digital_state[3] == e)
            if len(friend[0]) == 0:
                friend = np.where(self.digital_state[3] == add_players(self.index, self.index_second_agent))
            friend = (friend[0][0], friend[1][0])
            our_ghost_timer += self.digital_state[5][friend[0]][friend[1]]

        score = -enemy_food_returned - being_pacman * 10 + remaining_food - our_ghost_timer - distance_center - distance_attacking_enemies / 2
        return score

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
