import time

import gameengine as Engine
from action import Action
from actors.actor_random import ActorRandom
from gamestate import GameState

start_time = time.time()
runs = 0

while runs < 10000:
    runs += 1
    state: GameState = Engine.create_initial_state()
    actor: ActorRandom = ActorRandom()
    action: Action = Action(False)
    playable_combinations = None

    while not Engine.game_over(state):
        valid_action = False
        playable_combinations = Engine.score_combinations(state)

        while not valid_action:
            action: Action = actor.get_action(state, playable_combinations)
            valid_action = Engine.validate_action(state, action, playable_combinations)

        # states.append(utilities.serialize_state(state))
        # actions.append(utilities.serialize_action(action))
        state = Engine.step(state, action, playable_combinations)

    final_score = Engine.final_score(state)

# states.append(utilities.serialize_state(state))
# actions.append(utilities.serialize_action(action))

total_time = (time.time() - start_time)
print("average time: %s seconds" % (total_time / runs))
