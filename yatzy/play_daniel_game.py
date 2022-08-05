
from yatzy.mechanics import gameengine as Engine

from yatzy.mechanics.action import Action
from actors.actor_daniel import ActorDaniel
from yatzy.mechanics.gamestate import GameState


state: GameState = Engine.create_initial_state()
actor: ActorDaniel = ActorDaniel()
action: Action = Action(False)
playable_combinations = None

while not Engine.game_over(state):
    valid_action = False
    playable_combinations = Engine.score_combinations(state)

    while not valid_action:
        action: Action = actor.get_action(state, playable_combinations)
        valid_action = Engine.validate_action(state, action, playable_combinations)

    state = Engine.step(state, action, playable_combinations)
final_score = Engine.final_score(state)
