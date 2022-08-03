import gameengine as Engine
import input
import utilities

from action import Action
from gamestate import GameState

state: GameState = Engine.create_initial_state()

while not Engine.game_over(state):
    valid_action = False
    action: Action = Action(False)
    playable_combinations = Engine.score_combinations(state)
    utilities.print_game_state(state, playable_combinations)

    while not valid_action:
        action: Action = input.get_user_action()
        valid_action = Engine.validate_action(state, action, playable_combinations)
        print("valid action?", valid_action)

    state = Engine.step(state, action, playable_combinations)
