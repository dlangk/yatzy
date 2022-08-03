from yatzy.mechanics import gameengine as Engine, utilities

from yatzy.mechanics.action import Action
from yatzy.mechanics.gamestate import GameState
from actors.actor_user import ActorUser

state: GameState = Engine.create_initial_state()

actor = ActorUser()

while not Engine.game_over(state):
    valid_action = False
    action: Action = Action(False)
    playable_combinations = Engine.score_combinations(state)
    utilities.print_game_state(state, playable_combinations)

    while not valid_action:
        action: Action = actor.get_action(state, playable_combinations)
        valid_action = Engine.validate_action(state, action, playable_combinations)
        print("valid action?", valid_action)

    state = Engine.step(state, action, playable_combinations)
