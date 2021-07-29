import random
import engine


class Player:
    def __init__():
        pass

    def select_keepers(dices, state):
        keepers = []
        for dice in dices:
            keepers.append(random.choice([True, False]))
        return keepers

    def initial_roll():
        return engine.roll_dice(5)
