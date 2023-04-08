import time

from yatzy.mechanics import gameengine as Engine
from yatzy.mechanics.action import Action
from players.player_random import PlayerRandom
from yatzy.mechanics.gamestate import GameState

start_time = time.time()
runs = 0

while runs < 5000:
    runs += 1
    state: GameState = Engine.create_initial_state()
    actor: PlayerRandom = PlayerRandom()
    action: Action = Action(False)
    playable_combinations = None

    while not Engine.game_over(state):
        playable_combinations = Engine.score_combinations(state)

        valid_action = False
        while not valid_action:
            action: Action = actor.get_action(state, playable_combinations)
            valid_action = Engine.validate_action(state, action, playable_combinations)

        state = Engine.step(state, action, playable_combinations)

    final_score = Engine.final_score(state)

total_time = (time.time() - start_time)
print("final score: " + str(final_score))
print("average time: " + str(round((total_time / runs) * 1000, 4)))
