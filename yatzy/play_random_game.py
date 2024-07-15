import time

from yatzy.mechanics import gameengine as Engine, utilities
from yatzy.mechanics.action import Action
from players.player_random import Random
from yatzy.mechanics.gamestate import GameState

start_time = time.time()

state: GameState = Engine.create_initial_state()
player: Random = Random()
action: Action = Action(False)

while not Engine.game_over(state):
    valid_action = False
    while not valid_action:
        action: Action = player.get_action(state)
        valid_action = Engine.validate_action(state, action)
    state = Engine.step(state, action)

final_score = Engine.final_score(state)
total_time = (time.time() - start_time)

utilities.print_game_state(state)
print("time: " + str(round(total_time * 1000, 4)))
