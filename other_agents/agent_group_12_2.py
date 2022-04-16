from math import sqrt
from typing import Tuple

from capture import GameState
from captureAgents import CaptureAgent
import random
import time

from distanceCalculator import Distancer
from game import Directions

PLAYERS = []


def createTeam(
    firstIndex: int,
    secondIndex: int,
    isRed: bool,
    first: str = "DummyAgent",
    second: str = "DummyAgent",
):
    # The following line is an example only; feel free to change it.
    global PLAYERS
    PLAYERS = [eval(first)(firstIndex), eval(second)(secondIndex)]
    return PLAYERS


def get_direction(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> Directions:
    if pos1[0] == pos2[0] and pos1[1] == pos2[1]:
        return Directions.STOP
    else:
        if pos1[0] == pos2[0]:
            if pos1[1] < pos2[1]:
                return Directions.NORTH
            else:
                return Directions.SOUTH
        else:
            if pos1[0] < pos2[0]:
                return Directions.EAST
            else:
                return Directions.WEST


class DummyAgent(CaptureAgent):
    is_pacman: bool = True

    def registerInitialState(self, gameState: GameState):
        CaptureAgent.registerInitialState(self, gameState)

        self.distancer = Distancer(gameState.data.layout)

        self.reached_close_enemy = False
        self.positions = []

        if self.red:
            self.allies = gameState.getRedTeamIndices()
            self.enemies = gameState.getBlueTeamIndices()
        else:
            self.allies = gameState.getBlueTeamIndices()
            self.enemies = gameState.getRedTeamIndices()

        self.is_pacman = (
            self.index == (gameState.redTeam if self.red else gameState.blueTeam)[0]
        )

    def is_enemy_area(
        self, gameState: GameState, pos: Tuple[int, int], is_close: bool = False
    ) -> bool:
        if self.red:
            return pos[0] + (1 if is_close else 0) >= gameState.data.layout.width / 2
        else:
            return pos[0] - (1 if is_close else 0) <= gameState.data.layout.width / 2

    def is_enemy_food(self, gameState: GameState, pos: Tuple[int, int]) -> bool:
        return gameState.data.food.data[pos[0]][pos[1]] and self.is_enemy_area(
            gameState=gameState, pos=pos
        )

    def get_current_pos(self, gameState: GameState) -> Tuple[int, int]:
        current_pos = gameState.data.agentStates[self.index].configuration.pos
        return int(current_pos[0]), int(current_pos[1])

    def get_possible_positions(self, gameState: GameState, noisy_distance):
        """
        Returns possible positions found by DFS from a set of noisy distances.
        """
        initial_pos = self.get_current_pos(gameState=gameState)
        possible_positions = []

        parent = {}
        stack = []
        current_pos = None
        stack.append(initial_pos)

        current_dist = 0
        # DFS search.
        while stack:
            current_pos = stack.pop(-1)
            current_dist += 1

            if current_dist == noisy_distance:
                possible_positions.append(current_pos)
                current_dist -= 1
                break

            adjacents = [
                (current_pos[0] - 1, current_pos[1]),
                (current_pos[0] + 1, current_pos[1]),
                (current_pos[0], current_pos[1] - 1),
                (current_pos[0], current_pos[1] + 1),
            ]

            for adj in adjacents:
                if (
                    # Within bounds and not wall.
                    0 <= adj[0] <= gameState.data.layout.width
                    and 0 <= adj[1] <= gameState.data.layout.height
                    and not gameState.data.layout.walls.data[adj[0]][adj[1]]
                    and adj not in parent.keys()
                ):
                    parent[adj] = current_pos
                    stack.append(adj)
                else:
                    current_dist -= 1
        return possible_positions

    def pos_reachable(self, gameState: GameState, target_pos: Tuple[int, int]) -> bool:
        # Check if this agent can reach a position before it gets eaten by an enemy ghost.
        agent_pos = self.get_current_pos(gameState=gameState)
        agent_target_dist = self.distancer.getDistance(agent_pos, target_pos)
        for enemy in self.enemies:
            get_enemy_pos = gameState.getAgentPosition(enemy)
            enemy_positions = []
            if get_enemy_pos is None:
                # Enemy is not in sight, estimate position.
                noisy_enemy_dist = gameState.getAgentDistances()[
                    enemy
                ]  # Get noisy enemy distance.
                possible_enemy_positions = self.get_possible_positions(
                    gameState, noisy_enemy_dist
                )
                if not possible_enemy_positions:
                    # If no possible enemy position is found, return False.
                    # TODO: Return True instead?
                    return False
                enemy_positions = possible_enemy_positions
            else:
                # Exact enemy position is found.
                enemy_positions.append(get_enemy_pos)

            # Check if target can be reached before an enemy may reach it.
            for enemy_pos in enemy_positions:
                # if self.is_enemy_behind(gameState=gameState, target_pos=target_pos, enemy_pos=enemy_pos):
                #     # If enemy is behind it is not a threat.
                #     break
                enemy_target_dist = self.distancer.getDistance(enemy_pos, target_pos)
                if enemy_target_dist <= agent_target_dist:
                    # TODO: This check is not ideal since presumably the enemy chases the agent and not our target.
                    return False
        return True

    def is_target(self, gameState: GameState, pos: Tuple[int, int]):
        if self.is_pacman:
            if (
                self.is_close_to_ghost(
                    gameState=gameState, pos=self.get_current_pos(gameState=gameState),
                )
                or gameState.data.agentStates[self.index].numCarrying >= 10
                or sum(
                    [
                        sum(
                            [
                                1
                                for j, f in enumerate(fs)
                                if f
                                and self.is_enemy_area(gameState=gameState, pos=(i, j))
                            ]
                        )
                        for i, fs in enumerate(gameState.data.food.data)
                    ]
                )
                <= 2
            ):
                return not self.is_enemy_area(gameState=gameState, pos=pos)
            else:
                if any(
                    [
                        a.configuration.pos[0] == pos[0]
                        and a.configuration.pos[1] == pos[1]
                        and a.scaredTimer > 5
                        for i, a in enumerate(gameState.data.agentStates)
                        if a.configuration
                        and (
                            (self.red and i in gameState.blueTeam)
                            or (not self.red and i in gameState.redTeam)
                        )
                        and a.isPacman
                    ]
                ):
                    return True
                else:
                    return self.is_enemy_food(gameState=gameState, pos=pos)
        else:
            if not self.is_close_to_ghost(
                gameState=gameState,
                pos=self.get_current_pos(gameState=gameState),
                reversed=True,
            ):
                return self.is_enemy_area(gameState=gameState, pos=pos, is_close=True)
            else:
                if not self.reached_close_enemy and self.is_enemy_area(
                    gameState=gameState, pos=pos, is_close=True
                ):
                    return True
                elif self.is_enemy_area(gameState=gameState, pos=pos):
                    return False
                else:
                    return any(
                        [
                            a.configuration.pos[0] == pos[0]
                            and a.configuration.pos[1] == pos[1]
                            for i, a in enumerate(gameState.data.agentStates)
                            if a.configuration
                            and (
                                (self.red and i in gameState.blueTeam)
                                or (not self.red and i in gameState.redTeam)
                            )
                            and a.isPacman
                        ]
                    )

    def is_close_to_ghost(
        self, gameState: GameState, pos: Tuple[int, int], reversed: bool = False
    ):
        return any(
            [
                sqrt(
                    pow(abs(a.configuration.pos[0] - pos[0]), 2)
                    + pow(abs(a.configuration.pos[1] - pos[1]), 2)
                )
                < 4
                for i, a in enumerate(gameState.data.agentStates)
                if a.configuration
                and (
                    (self.red and i in gameState.blueTeam)
                    or (not self.red and i in gameState.redTeam)
                )
                and (a.isPacman if reversed else not a.isPacman)
            ]
        )

    def chooseAction(self, gameState: GameState):
        initial_pos = self.get_current_pos(gameState=gameState)
        actions = gameState.getLegalActions(self.index)

        if (
            gameState.data.score < 0
            and len(self.positions) > 50
            and len(set(self.positions[-50:])) < 2
        ):
            self.is_pacman = not self.is_pacman
            self.positions = []
            other_player = [p for p in PLAYERS if p.index != self.index][0]
            other_player.is_pacman = not other_player.is_pacman
            other_player.positions = []

        parent = {}
        queue = []
        current_pos = None
        queue.append(initial_pos)

        if self.is_enemy_area(gameState=gameState, pos=initial_pos, is_close=True):
            self.reached_close_enemy = True

        while queue:
            current_pos = queue.pop(0)
            if self.is_target(gameState=gameState, pos=current_pos):
                break

            adjacents = [
                (current_pos[0] - 1, current_pos[1]),
                (current_pos[0] + 1, current_pos[1]),
                (current_pos[0], current_pos[1] - 1),
                (current_pos[0], current_pos[1] + 1),
            ]

            for adj in adjacents:
                if (
                    0 <= adj[0] <= gameState.data.layout.width
                    and 0 <= adj[1] <= gameState.data.layout.height
                    and not gameState.data.layout.walls.data[adj[0]][adj[1]]
                    and adj not in parent.keys()
                    and (
                        self.is_pacman
                        or not self.is_enemy_area(gameState=gameState, pos=adj)
                    )
                    and not self.pos_reachable(gameState=gameState, target_pos=adj)
                ):
                    parent[adj] = current_pos
                    queue.append(adj)

        if current_pos:
            while current_pos in parent.keys() and parent[current_pos] != initial_pos:
                current_pos = parent[current_pos]
            self.positions.append(current_pos)
            return get_direction(initial_pos, current_pos)
        return random.choice(actions)
