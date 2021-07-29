import engine
import brain
import json

def ppj(parsed_json):
    print(json.dumps(parsed_json, indent=4, sort_keys=True))

state = engine.initialize_game(["brain1"])
engine.print_state(state)

turns = 0
max_turns = 0
while turns <= max_turns:
    dices = engine.get_dice_state(state)
    if engine.get_active_player_roll(state) == 0:
        state = engine.start_turn(state)
    # player now needs to decide what to keep
    dices = engine.get_dice_state(state)
    print("new dice state", dices)
    options = engine.get_options(dices)
    ppj(options)
    engine.print_state(state)
    turns += 1

