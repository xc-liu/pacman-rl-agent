import time
from Mini_max_game import Mini_Max_Game

class Node:
    def __init__(self, game_state, parent=None, move=None, depth=0, agent=0, player=0):
        self.children = []
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.depth = depth
        self.agent_idx = agent
        self.player = player

    def compute_children(self, l=None):
        # init_time = time.time()
        new_idx = self.agent_idx + 1 if self.agent_idx < 3 else 0
        if l is None:
            possible_moves = self.game_state.get_legal_actions(self.agent_idx)
        else:
            possible_moves = l
        for p in possible_moves:
            updated_game_state = self.game_state.get_next_game(p, self.agent_idx)
            new_node = self.__class__(updated_game_state, self, p, self.depth + 1, new_idx,
                                      1 - self.player)
            self.children.append(new_node)
            # break
        # print(time.time() - init_time)
        # exit()
        return self.children


class MiniMax():
    def __init__(self, player_idx=None, enemy_idx=None, agent=None):
        self.time_limit = 200
        self.time_init = 0
        self.player_idx = player_idx
        self.enemy_idx = enemy_idx
        self.agent = agent

    def update_players(self):
        self.offensive_players = [self.player_idx[0], self.enemy_idx[0]]
        self.defensive_players = [self.player_idx[1], self.enemy_idx[1]]

    def alpha_beta(self, node, d, a, B, player):
        to_expand = node.compute_children()
        if d == 0 or len(to_expand) == 0:
            if node.agent_idx in self.offensive_players:
                v = node.game_state.get_offensive_heuristic(node.agent_idx)
            else:
                v = node.game_state.get_defensive_heuristic(node.agent_idx)
        else:
            if player == 0:
                v = -99999
                for child in to_expand:
                    v = max(v, self.alpha_beta(child, d - 1, a, B, 1))
                    a = max(a, v)
                    if B <= a or ((time.time() - self.time_init) * 1000) >= self.time_limit: break
            else:
                v = 99999
                for child in to_expand:
                    v = min(v, self.alpha_beta(child, d - 1, a, B, 0))
                    B = min(B, v)
                    if B <= a or ((time.time() - self.time_init) * 1000) >= self.time_limit: break
        return v

    def search_best_move(self, game_state, agent_idx, l):
        game_state.visualise_state()
        time.sleep(1)
        node = Node(game_state, agent=agent_idx, player=0)
        to_expand = node.compute_children(l)
        d = 1
        best_move = "Stop"
        best_ev = -99999
        self.time_init = time.time()
        while (time.time() - self.time_init) * 1000 <= self.time_limit:
            child_evs = []
            for child in to_expand:
                child_evs.append(self.alpha_beta(child, d, -99999, 99999, child.player))
            best_child = max(child_evs)
            if best_child > best_ev:
                best_ev = best_child
                best_move = to_expand[child_evs.index(best_child)].move
            d += 1
        return best_move

