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


class stateRepresentation:
    def __init__(self, agent, gameState, index, red):
        self.agent = agent
        self.gameState = gameState
        self.index = index
        self.red = red
        self.next_pos = {}
        self.dist_history = {}
        self.last_enemy_pos = {}
        self.digital_state = self.initialise_digital_state(gameState)
        self.initialise_next_pos(steps=2)
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
                    elif idx in [0, 2]:
                        digital_state[3][i][j] = idx + 1
                    else:
                        digital_state[3][i][j] = idx - 1

                    if digital_state[3][i][j] % 2 == 0:
                        self.last_enemy_pos[int(digital_state[3][i][j])] = (j, len(layout) - i - 1)

        myPos = gameState.getAgentState(self.index).getPosition()
        for i in [2, 4]:
            self.dist_history[i] = [self.agent.distancer.getDistance(pos1=self.last_enemy_pos[i], pos2=myPos)]

        return digital_state

    def initialise_next_pos(self, steps=2):
        h = len(self.digital_state[0])
        w = len(self.digital_state[0][0])
        for i in range(h):
            for j in range(w):
                if self.digital_state[0][h - 1 - i][j] == 0:
                    pos = (i, j)
                    self.next_pos[(pos[1], pos[0])] = []
                    possible_pos = [(i + p, j + q)
                                    for p in range(-steps, steps + 1)
                                    for q in range(-steps, steps + 1)
                                    if 0 <= i + p < h and 0 <= j + q < w]
                    for p in possible_pos:
                        if self.digital_state[0][h - 1 - p[0]][p[1]] == 0:
                            if self.agent.getMazeDistance((pos[1], pos[0]), (p[1], p[0])) <= steps:
                                self.next_pos[(pos[1], pos[0])].append((p[1], p[0]))

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
                    idx2 = height - i - 1

                if r_food[idx1][idx2] or b_food[idx1][idx2]:
                    self.digital_state[1][i][j] = 1

        # if the current food layer is different from the previous one, update the previous position of the enemy
        my_food_change = my_prev_food - self.digital_state[1, :, :int(width / 2)]
        change_food = np.nonzero(my_food_change)
        change_pos = [(b, height - a - 1) for a in change_food[0] for b in change_food[1]]

        # update capsule
        if self.red:
            if len(r_capsule) > 0: self.digital_state[2][height - 1 - r_capsule[0][1]][r_capsule[0][0]] = 1
            if len(b_capsule) > 0: self.digital_state[2][height - 1 - b_capsule[0][1]][b_capsule[0][0]] = 1
        else:
            if len(r_capsule) > 0:
                self.digital_state[2][r_capsule[0][1]][width - 1 - r_capsule[0][0]] = 1
            if len(b_capsule) > 0:
                self.digital_state[2][b_capsule[0][1]][width - 1 - b_capsule[0][0]] = 1

        my_capsule_change = my_prev_capsule - self.digital_state[2, :, :int(width / 2)]
        change_capsule = np.nonzero(my_capsule_change)
        change_pos += [(b, height - a - 1) for a in change_capsule[0] for b in change_capsule[1]]

        # update player states
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
                idx = idx + 1

            if enemy:

                if pos is not None:
                    myPos = gameState.getAgentState(self.index).getPosition()
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
                            myPos = gameState.getAgentState(self.index).getPosition()
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
                        myPos = gameState.getAgentState(self.index).getPosition()
                        self.dist_history[idx].append(self.agent.distancer.getDistance(myPos, pos))
                        # self.dist_history[idx] = [self.agent.distancer.getDistance(myPos, pos)]
                        self.last_enemy_pos[idx] = pos

                    if not changed:
                        pos = self.computeOpponentPosition(idx, original_idx)

            if not self.red:
                pos = [width - 1 - pos[0], height - 1 - pos[1]]

            if self.digital_state[3][height - 1 - int(pos[1])][int(pos[0])] == 0:
                self.digital_state[3][height - 1 - int(pos[1])][int(pos[0])] = idx
            else:
                self.digital_state[3][height - 1 - int(pos[1])][int(pos[0])] += idx + 1

            # digital_state[4][height - int(pos[1])][int(pos[0])] = actions_idx[direction]
            self.digital_state[4][height - 1 - int(pos[1])][int(pos[0])] += food_carrying if pacman else 0
            self.digital_state[5][height - 1 - int(pos[1])][int(pos[0])] += scared_timer

        return self.digital_state

    def computeOpponentPosition(self, idx, original_idx):
        noisy_dist = np.clip(self.gameState.getAgentDistances()[original_idx], a_min=5, a_max=None)

        # use Kalman filter to correct the noisy distance
        self.dist_history[idx].append(noisy_dist)
        corrected_dist = np.clip(kalman(self.dist_history[idx], 0.01, 0.01), a_min=5, a_max=None)
        # corrected_dist = noisy_dist

        # sample around the last enemy state, find the one with closest distance to corrected_dist
        possible_enemy_pos = self.next_pos[(int(self.last_enemy_pos[idx][0]), int(self.last_enemy_pos[idx][1]))]
        possible_distances = []
        myPos = self.gameState.getAgentState(self.index).getPosition()
        for p in possible_enemy_pos:
            possible_distances.append(self.agent.distancer.getDistance(myPos, p))
        self.last_enemy_pos[idx] = possible_enemy_pos[find_nearest(possible_distances, corrected_dist)]
        return self.last_enemy_pos[idx]

    def get_state_info(self):
        return self.digital_state, self.score, self.time_left

    def visualise_state(self):
        st = ""
        food_carrying = {}
        scared = {}
        for i in range(len(self.digital_state[0])):
            for j in range(len(self.digital_state[0][0])):

                if self.digital_state[3][i][j] != 0:
                    st += str(int(self.digital_state[3][i][j]))
                    if self.digital_state[4][i][j] != 0:
                        food_carrying[int(self.digital_state[3][i][j])] = self.digital_state[4][i][j]
                    if self.digital_state[5][i][j] != 0:
                        scared[int(self.digital_state[3][i][j])] = self.digital_state[5][i][j]
                elif self.digital_state[0][i][j] == 1:
                        st += "%"
                elif self.digital_state[1][i][j] == 1:
                    st += "."
                elif self.digital_state[2][i][j] == 1:
                    st += "o"
                else:
                    st += " "
            st += "\n"
        st = st[:-1]

        info = "Time left %s, score %s. " % (self.time_left, self.score)
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

    def visualise_one_layer(self, layer):
        st = ""
        for i in range(len(layer)):
            for j in range(len(layer[0])):
                st += str(int(layer[i][j])) + " "
            st += "\n"
        print(st)
