import pygame
from Board import Board
from Player import *
import PygameFunctions as PF
from constants import *
from Board import actions_taken


SELF_PLAY = True  # whether or not a human will be playing
player_role = "Government"  # Valid options are "Government" and "Zombie"
# Create the game board
GameBoard = Board((ROWS, COLUMNS), player_role)
GameBoard.populate()

# Initialize variables
running = True
take_action = []
playerMoved = False

enemy_player = None
if player_role == "Government":
    enemy_player = ZombieMinimaxPlayer()
    ai_player = GovernmentMinimaxPlayer()
    dummy_player = GovernmentPlayer()
else:
    enemy_player = GovernmentMinimaxPlayer()
    ai_player = ZombieMinimaxPlayer()
    dummy_player = ZombiePlayer()

PF.initScreen(GameBoard)


while running:
    P = PF.run(GameBoard)

    if not playerMoved:
        GameBoard.telemetry = "Your move!"
        if SELF_PLAY:
            if not dummy_player.get_move(GameBoard)[0]:
                PF.csv_update("data.csv", GameBoard.resources.getCosts(), actions_taken)
                PF.display_lose_screen()
                running = False
                continue
            # Event Handling
            for event in P:
                if event.type == pygame.MOUSEBUTTONUP:
                    x, y = pygame.mouse.get_pos()
                    action = PF.get_action(GameBoard, x, y)
                    if (
                        action == "cure"
                        or action == "vaccinate"
                        or action == "wall"
                        or action == "bite"
                    ):
                        # only allow healing by itself (prevents things like ['move', (4, 1), 'cure'])
                        if len(take_action) == 0:
                            take_action.append(action)
                    elif action == "reset move":
                        take_action = []
                    elif action is not None:
                        idx = GameBoard.toIndex(action)
                        # action is a coordinate
                        if idx < (GameBoard.rows * GameBoard.columns) and idx > -1:
                            if "move" not in take_action and len(take_action) == 0:
                                # make sure that the space is not an empty space or a space of the opposite team
                                # since cannot start a move from those invalid spaces
                                if (
                                    GameBoard.States[idx].person is not None
                                    and GameBoard.States[idx].person.isZombie
                                    == ROLE_TO_ROLE_BOOLEAN[player_role]
                                ):
                                    take_action.append("move")
                                else:
                                    continue

                            # don't allow duplicate cells
                            if action not in take_action:
                                take_action.append(action)
                if event.type == pygame.QUIT:
                    PF.csv_update("data.csv", GameBoard.resources.getCosts(), actions_taken)
                    running = False

            PF.display_cur_move(take_action)

            # Action handling
            if len(take_action) > 1:
                if take_action[0] == "move":
                    if len(take_action) > 2:
                        directionToMove = PF.direction(take_action[1], take_action[2])
                        result = GameBoard.actionToFunction[directionToMove](
                            take_action[1]
                        )
                        if result[0] is not False:
                            playerMoved = True
                            PF.record_actions("movesMade", actions_taken)
                        take_action = []

                elif (
                    take_action[0] == "cure"
                    or take_action[0] == "vaccinate"
                    or take_action[0] == "bite"
                    or take_action[0] == "wall"
                ):
                    result = GameBoard.actionToFunction[take_action[0]](take_action[1])
                    if result[0] is not False:
                        playerMoved = True
                    take_action = []

        # ai player as player 1
        else:
            action, move_coord = ai_player.get_move(GameBoard)
            if not action:
                PF.csv_update("data.csv", GameBoard.resources.getCosts(), actions_taken)
                running = False
                PF.display_lose_screen()
                continue

            # Select the destination coordinates
            # print(f"choosing to go with {action} at {move_coord}")
            GameBoard.telemetry = (
                f"the AI chose to {action}"  # reset telemetry and add AI move
            )

            # Implement the selected action
            GameBoard.actionToFunction[action](move_coord)
            playerMoved = True
            continue

    # Computer turn
    else:
        playerMoved = False
        take_action = []
        action, move_coord = enemy_player.get_move(GameBoard)

        if not action:
            PF.csv_update("data.csv", GameBoard.resources.getCosts(), actions_taken)
            running = False
            PF.display_win_screen()
            continue

        # print(f"choosing to go with {action} at {move_coord}")
        # Implement the selected action
        GameBoard.actionToFunction[action](move_coord)
        # Update the board's states
        GameBoard.update()

    # Update the display
    PF.display_telemetry(GameBoard.telemetry)
    pygame.display.update()

    pygame.time.wait(200)
