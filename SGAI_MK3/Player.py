from abc import abstractmethod
from typing import Dict, Tuple, List, Union

import numpy as np
from Board import Board
from constants import *
import random as rd
import math

# AI STUFF
import numpy as np
import tensorflow as tf
from keras import layers
import keras


class GovernmentEnvironment:
    ACTION_MAPPINGS = {
        0: "moveUp",
        1: "moveDown",
        2: "moveLeft",
        3: "moveRight",
        4: "wallUp",
        5: "wallDown",
        6: "wallLeft",
        7: "wallRight",
        8: "vaccinate",
        9: "cureUp",
        10: "cureDown",
        11: "cureLeft",
        12: "cureRight",
    }
    ACTION_NAMES = [
        "moveUp",
        "moveDown",
        "moveLeft",
        "moveRight",
        "wallUp",
        "wallDown",
        "wallLeft",
        "wallRight",
        "vaccinate",
        "cureUp",
        "cureDown",
        "cureLeft",
        "cureRight",
    ]
    ACTION_SPACE = len(ACTION_MAPPINGS.keys())
    SIZE = (6, 6)
    OBSERVATION_SPACE = 40

    def __init__(self, max_timesteps: int = 300, logdir: str = "", run_name="") -> None:
        self.max_timesteps = max_timesteps
        self.reset()
        self.total_timesteps = 0
        self.total_invalid_moves = 0
        self.writer = None
        if logdir != "" and run_name != "":
            self.writer = tf.summary.create_file_writer(f"{logdir}/{run_name}")

    def reset(self):
        self.board = Board(GovernmentEnvironment.SIZE, "Government")
        num_people = rd.randint(7, 12)
        self.board.populate(num_people=num_people, num_zombies=num_people - 1)
        self.enemyPlayer = ZombiePlayer()
        self.done = False

        # coordinates of the first Government player
        self.agentPosition = self.board.indexOf(False)
        if self.agentPosition == -1 or not self.board.is_move_possible_at(
            self.agentPosition
        ):
            self.reset()

        # useful for metrics
        self.max_number_of_government = 1
        self.episode_invalid_actions = 0
        self.episode_reward = 0
        self.episode_timesteps = 0
        return self._get_obs()

    def copy(self, board: Board, position: int):
        self.board = board.clone(board.States, board.player_role)
        self.agentPosition = position

    def get_invalid_action_mask(self):
        action_mask = [True for name in GovernmentEnvironment.ACTION_NAMES]
        clone = self.board.clone(self.board.States, self.board.player_role)
        coord = self.board.toCoord(self.agentPosition)
        for idx in range(len(GovernmentEnvironment.ACTION_NAMES)):
            action_name = GovernmentEnvironment.ACTION_NAMES[idx]

            if "move" in action_name:
                valid, new_pos = clone.actionToFunction[action_name](coord)
            elif "vaccinate" in action_name:
                valid, _ = clone.actionToFunction[action_name](coord)
            elif "cure" in action_name:
                dest_coord = list(coord)
                if action_name == "cureUp":
                    dest_coord[1] -= 1
                elif action_name == "cureDown":
                    dest_coord[1] += 1
                elif action_name == "cureRight":
                    dest_coord[0] += 1
                else:
                    dest_coord[0] -= 1
                valid, _ = clone.actionToFunction["cure"](dest_coord)
            else:  # wall variation
                dest_coord = list(coord)
                if action_name == "wallUp":
                    dest_coord[1] -= 1
                elif action_name == "wallDown":
                    dest_coord[1] += 1
                elif action_name == "wallRight":
                    dest_coord[0] += 1
                else:
                    dest_coord[0] -= 1
                valid, _ = clone.actionToFunction["wall"](dest_coord)

            if valid:
                # re-clone the board
                clone = self.board.clone(self.board.States, self.board.player_role)
            else:
                action_mask[idx] = False

        return action_mask

    def get_action(self, action: int):
        action_name = GovernmentEnvironment.ACTION_MAPPINGS[action]
        if "move" in action_name:
            return action_name, self.board.toCoord(self.agentPosition)
        elif "vaccinate" in action_name:
            return action_name, self.board.toCoord(self.agentPosition)
        elif "cure" in action_name:
            dest_coord = list(self.board.toCoord(self.agentPosition))
            if action_name == "cureUp":
                dest_coord[1] -= 1
            elif action_name == "cureDown":
                dest_coord[1] += 1
            elif action_name == "cureRight":
                dest_coord[0] += 1
            else:
                dest_coord[0] -= 1
            return "cure", dest_coord
        else:  # wall variation
            dest_coord = list(self.board.toCoord(self.agentPosition))
            if action_name == "wallUp":
                dest_coord[1] -= 1
            elif action_name == "wallDown":
                dest_coord[1] += 1
            elif action_name == "wallRight":
                dest_coord[0] += 1
            else:
                dest_coord[0] -= 1
            return "wall", dest_coord

    def _get_obs(self):
        """
        Is based off the assumption that 5 is not in the returned board.
        Uses 5 as the key for current position.
        """
        AGENT_POSITION_CONSTANT = 5
        ret = self.board.get_board()
        ret[self.agentPosition] = AGENT_POSITION_CONSTANT

        # add resources and prices
        ret.append(self.board.resources.resources)
        ret.append(self.board.resources.costs["cure"])
        ret.append(self.board.resources.costs["vaccinate"])
        ret.append(self.board.resources.costs["wall"])

        return np.array(ret, dtype=np.float32)


NUM_ACTIONS = 13


def mlp(x, sizes, activation=tf.tanh, output_activation=None):
    # Build a feedforward neural network
    for size in sizes[:-1]:
        x = layers.Dense(units=size, activation=activation)(x)
    return layers.Dense(units=sizes[-1], activation=output_activation)(x)


# Sample action from actor
@tf.function
def apply_invalid_mask(logits, action_mask):
    # pass in logits; this would be before doing logprobabilities
    # applies an invalid action mask
    invalid_values = tf.constant([[tf.float32.min] * NUM_ACTIONS], dtype=tf.float32)

    logits = tf.where(action_mask, logits, invalid_values)
    return logits


@tf.function
def get_best_action(observation, invalid_action_mask):
    logits = actor(observation, training=False)
    logits = apply_invalid_mask(logits, invalid_action_mask)
    action = tf.argmax(tf.squeeze(logits))

    # prioritize curing
    if logits[0][9] != tf.float32.min:
        action = tf.constant([9], dtype=tf.int64)
    elif logits[0][10] != tf.float32.min:
        action = tf.constant([10], dtype=tf.int64)
    elif logits[0][11] != tf.float32.min:
        action = tf.constant([11], dtype=tf.int64)
    elif logits[0][12] != tf.float32.min:
        action = tf.constant([12], dtype=tf.int64)

    return logits, action


observation_input = keras.Input(shape=(40,), dtype=tf.float32)
hidden_sizes = (256, 256)
logits = mlp(observation_input, list(hidden_sizes) + [NUM_ACTIONS], tf.tanh, None)
actor = keras.Model(inputs=observation_input, outputs=logits)
value = tf.squeeze(
    mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1
)
critic = keras.Model(inputs=observation_input, outputs=value)

actor.load_weights("./gov_actor_v2_weights")
critic.load_weights("./gov_critic_v2_weights")


class Player:
    """
    Base class for a player, takes
    random actions
    """

    def __init__(self, player_name, verbose=False) -> None:
        self.player_name = player_name
        self.verbose = verbose
        self.possible_actions = [
            ACTION_SPACE[i]
            for i in range(len(ACTION_SPACE))
            if (ACTION_SPACE[i] != "bite" and self.player_name == "Government")
            or (
                ACTION_SPACE[i] != "cure"
                and ACTION_SPACE[i] != "vaccinate"
                and ACTION_SPACE[i] != "wall"
                and self.player_name == "Zombie"
            )
        ]

    def get_all_possible_actions(
        self, board: Board
    ) -> Dict[str, List[Tuple[int, int]]]:

        ret = {}
        for action in self.possible_actions:
            ret[action] = board.get_possible_moves(
                action, "Zombie" if self.player_name == "Zombie" else "Government"
            )
        return ret

    def get_move(self, board: Board) -> Tuple[str, Tuple[int, int]]:
        # Make a list of all possible actions that the computer can take
        possible_actions = self.possible_actions.copy()
        possible_move_coords = []
        while len(possible_move_coords) == 0 and len(possible_actions) != 0:
            if self.verbose:
                print("possible actions are", possible_actions)
            # if "bite" in possible_actions:
            #    action = "bite"
            #    possible_actions.remove("bite")
            else:
                action = possible_actions.pop(rd.randint(0, len(possible_actions) - 1))
            possible_move_coords = board.get_possible_moves(
                action, "Zombie" if self.player_name == "Zombie" else "Government"
            )

        # no valid moves, player wins
        if len(possible_actions) == 0 and len(possible_move_coords) == 0:
            if self.verbose:
                print("no possible moves for the computer")
                if self.player_name == "Zombie":
                    print(
                        f"The government ended with {board.resources.resources} resources"
                    )
                    print(
                        f"The price of vaccination was {board.resources.costs['vaccinate']} and the price of curing was {board.resources.costs['cure']}"
                    )
            return False, None

        # Select the destination coordinates
        move_coord = rd.choice(possible_move_coords)
        if self.verbose:
            print(f"choosing to go with {action} at {move_coord}")
        return (action, move_coord)


class GovernmentPlayer(Player):
    """
    Plays as the government
    """

    def __init__(self, verbose=False) -> None:
        super().__init__("Government", verbose)


class ZombiePlayer(Player):
    """
    Plays as the zombies
    """

    def __init__(self, verbose=False) -> None:
        super().__init__("Zombie", verbose)


class GovernmentAIPlayer(GovernmentPlayer):
    """
    Will be a smarter version of the Human Player
    """

    def __init__(self) -> None:
        super().__init__()

    def get_move(self, board: Board) -> Tuple[str, Tuple[int, int]]:
        if not GovernmentPlayer().get_move(board)[0]:
            return None, None
        max_val = float("-inf")
        best_idx = -1
        best_action = -1
        temp_env = GovernmentEnvironment()
        for idx in range(len(board.States)):
            if (
                board.States[idx].person is not None
                and board.States[idx].person.isZombie is False
            ):
                print("happening here at", board.toCoord(idx))
                temp_env.copy(board, idx)
                obs = temp_env._get_obs()
                obs = np.reshape(obs, (1, 40))
                mask = tf.cast([temp_env.get_invalid_action_mask()], dtype=tf.bool)
                logits, action = get_best_action(obs, mask)
                action = int(action.numpy())
                value = critic(obs, training=False)
                print(value)
                value = value.numpy()[0]

                if value > max_val:
                    max_val = value
                    best_idx = idx
                    best_action = action

        print("best idx is", best_idx)
        temp_env.copy(board, best_idx)
        action_name, coord = temp_env.get_action(best_action)
        print(f"action is {action_name} and coord is {coord}")
        return action_name, coord


class ZombieAIPlayer(ZombiePlayer):
    """
    Will be a smarter version of the Zombie Player
    """

    def __init__(self) -> None:
        super().__init__()

    def get_move(self, board: Board) -> Tuple[str, Tuple[int, int]]:
        print("zombie get move ai")
        best_action = None
        best_value = float("-inf")
        best_pos = None
        for i in range(36):
            p = board.States[i].person
            if p is not None and p.isZombie:
                action, pos, value = p.get_best_move(board, i)
                if value > best_value:
                    best_action = action
                    best_value = value
                    best_pos = pos
        if best_pos is None:
            print("using zombie minimax player")
            action, best_pos = ZombieMinimaxPlayer().get_move(board)
        return best_action, best_pos


class MiniMaxPlayer(Player):
    def __init__(self, player_name, verbose=False, lookahead: int = 3) -> None:
        super().__init__(player_name, verbose)
        self.lookahead = 3

    @abstractmethod
    def _get_value(self, board: Board) -> float:
        raise NotImplementedError(
            "_get_value must be implemented in a subclass of MinimaxPlayer"
        )

    def _minimax_val(
        self,
        board: Board,
        self_turn: bool,
        depth: int,
        alpha: Union[float, int],
        beta: Union[float, int],
    ) -> int:
        """
        Returns the highest value possible from a state
        If depth is 0, returns the board's current value.
        If not self turn, returns the least possible value possible from the board
        @param alpha - the best case found so far for the maximizing player (highest reachable val)
        @param beta - the best case found so far for the minimizing player (lowest reachable val)
        """
        if depth == 0:
            return self._get_value(board)

        mappings = (self if self_turn else self.enemyPlayer).get_all_possible_actions(
            board
        )
        ret = float("-inf" if self_turn else "inf")
        for action, lst in mappings.items():
            for coord in lst:
                clone = board.clone(board.States.copy(), board.player_role)
                clone.actionToFunction[action](coord)
                value = self._minimax_val(clone, not self_turn, depth - 1, alpha, beta)
                if self_turn:
                    ret = max(ret, value)
                    alpha = max(alpha, ret)

                    # alpha beta pruning
                    if beta <= alpha:
                        return ret
                else:
                    ret = min(value, ret)
                    beta = min(beta, ret)

                    # alpha beta pruning
                    if alpha <= beta:
                        return ret

        return ret

    def get_moves(self, board: Board) -> Dict[str, List[Tuple[int, int]]]:
        mappings = self.get_all_possible_actions(board)
        max_val = float("-inf")
        action_coordlists = {}
        for action, lst in mappings.items():
            for coord in lst:
                clone = board.clone(board.States.copy(), board.player_role)
                clone.actionToFunction[action](coord)
                val = self._minimax_val(
                    clone, False, self.lookahead - 1, float("-inf"), float("inf")
                )
                if val > max_val:
                    action_coordlists.clear()
                    max_val = val
                    action_coordlists[action] = [coord]
                if val == max_val:
                    if action not in action_coordlists:
                        action_coordlists[action] = [coord]
                    else:
                        action_coordlists[action].append(coord)
        return action_coordlists


class GovernmentMinimaxPlayer(MiniMaxPlayer):
    """
    Government player, minimaxed
    """

    def __init__(self, lookahead: int = 3) -> None:
        """
        Initializes a GovernmentMinimaxPlayer
        @param lookahead How far to look ahead
        """
        super().__init__("Government", lookahead=lookahead)
        self.enemyPlayer = ZombiePlayer()

    def _get_value(self, board: Board) -> int:
        """
        Returns how valuable a board is.
        Rewards based on # of people on the board
        """
        zombie_count = 0
        zombie_x = 0
        zombie_y = 0

        people_count = 0
        people_x = 0
        people_y = 0
        for idx in range(len(board.States)):
            state = board.States[idx]
            if state.person is not None:
                x, y = board.toCoord(idx)
                if state.person.isZombie:
                    zombie_x += x
                    zombie_y += y
                    zombie_count += 1
                else:
                    people_x += x
                    people_y += y
                    people_count += 1

        if people_count != 0 and zombie_count != 0:
            people_x /= people_count
            people_y /= people_count
            zombie_x /= zombie_count
            zombie_y /= zombie_count
            return people_count + 0.75 * (
                1 + np.sqrt((people_x - zombie_x) ** 2 + (people_y - zombie_y) ** 2)
            )

        return people_count

    def get_move(self, board: Board) -> Tuple[str, Tuple[int, int]]:
        action_coordlists = self.get_moves(board)

        if len(action_coordlists) == 0:
            return None, (-1, -1)

        if "cure" in action_coordlists:
            action = "cure"
        elif "vaccinate" in action_coordlists:
            action = "vaccinate"
        else:
            action = rd.choice(tuple(action_coordlists.keys()))
        return action, rd.choice(action_coordlists[action])


class ZombieMinimaxPlayer(MiniMaxPlayer):
    """
    Zombie player, minimaxed
    """

    def __init__(self, lookahead: int = 3) -> None:
        """
        Initializes a ZombieMinimaxPlayer
        @param lookahead How far to look ahead
        """
        super().__init__("Zombie", lookahead=lookahead)
        self.enemyPlayer = GovernmentPlayer()

    def _get_value(self, board: Board) -> int:
        """
        Returns how valuable a board is.
        Rewards based on # of zombies on the board
        and distance between people centroid and zombie centroid
        """
        zombie_count = 0
        zombie_x = 0
        zombie_y = 0

        people_count = 0
        people_x = 0
        people_y = 0
        for idx in range(len(board.States)):
            state = board.States[idx]
            if state.person is not None:
                x, y = board.toCoord(idx)
                if state.person.isZombie:
                    zombie_x += x
                    zombie_y += y
                    zombie_count += 1
                else:
                    people_x += x
                    people_y += y
                    people_count += 1

        if people_count != 0 and zombie_count != 0:
            people_x /= people_count
            people_y /= people_count
            zombie_x /= zombie_count
            zombie_y /= zombie_count
            return zombie_count + 0.75 / (
                1 + np.sqrt((people_x - zombie_x) ** 2 + (people_y - zombie_y) ** 2)
            )
        return zombie_count

    def get_move(self, board: Board) -> Tuple[str, Tuple[int, int]]:
        action_coordlists = self.get_moves(board)

        if len(action_coordlists) == 0:
            return None, (-1, -1)

        if "bite" in action_coordlists:
            action = "bite"
        else:
            action = rd.choice(tuple(action_coordlists.keys()))
        return action, rd.choice(action_coordlists[action])
