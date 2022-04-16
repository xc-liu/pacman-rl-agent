from math import sqrt
from typing import Tuple

from capture import GameState
from captureAgents import CaptureAgent
import random, time

from game import Directions


def createTeam(
    firstIndex: int,
    secondIndex: int,
    isRed: bool,
    first: str = "DummyAgent",
    second: str = "DummyAgent",
):
    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


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

        self.is_pacman = (
            self.index == (gameState.redTeam if self.red else gameState.blueTeam)[0]
        )

    def is_enemy_area(self, gameState: GameState, pos: Tuple[int, int]) -> bool:
        if self.red:
            return pos[0] >= gameState.data.layout.width / 2
        else:
            return pos[0] <= gameState.data.layout.width / 2

    def is_enemy_food(self, gameState: GameState, pos: Tuple[int, int]) -> bool:
        return gameState.data.food.data[pos[0]][pos[1]] and self.is_enemy_area(
            gameState=gameState, pos=pos
        )

    def get_current_pos(self, gameState: GameState) -> Tuple[int, int]:
        current_pos = gameState.data.agentStates[self.index].configuration.pos
        return int(current_pos[0]), int(current_pos[1])

    def is_target(self, gameState: GameState, pos: Tuple[int, int]):
        if self.is_pacman:
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
                if (
                    self.is_close_to_ghost(
                        gameState=gameState,
                        pos=self.get_current_pos(gameState=gameState),
                    )
                    or gameState.data.agentStates[self.index].numCarrying >= 9
                ):
                    return not self.is_enemy_area(gameState=gameState, pos=pos)
                else:
                    return self.is_enemy_food(gameState=gameState, pos=pos)
        else:
            if self.is_enemy_area(gameState=gameState, pos=pos):
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

    def is_close_to_ghost(self, gameState: GameState, pos: Tuple[int, int]):
        return any(
            [
                sqrt(
                    pow(abs(a.configuration.pos[0] - pos[0]), 2)
                    + pow(abs(a.configuration.pos[1] - pos[1]), 2)
                )
                < 3
                for i, a in enumerate(gameState.data.agentStates)
                if a.configuration
                and (
                    (self.red and i in gameState.blueTeam)
                    or (not self.red and i in gameState.redTeam)
                )
                and not a.isPacman
            ]
        )

    def chooseAction(self, gameState: GameState):
        time.sleep(0.025)
        initial_pos = self.get_current_pos(gameState=gameState)
        actions = gameState.getLegalActions(self.index)

        parent = {}
        queue = []
        current_pos = None
        queue.append(initial_pos)

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
                ):
                    parent[adj] = current_pos
                    queue.append(adj)

        if current_pos:
            while current_pos in parent.keys() and parent[current_pos] != initial_pos:
                current_pos = parent[current_pos]
            return get_direction(initial_pos, current_pos)
        return random.choice(actions)
