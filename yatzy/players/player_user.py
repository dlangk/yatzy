from yatzy.mechanics.gamestate import GameState
from yatzy.mechanics.action import Action


class PlayerUser:

    def __init__(self):
        pass

    def get_action(self, state: GameState, playable_combinations) -> Action:
        score_input = input("Do you want to score? (y/n)\n")
        score = (score_input == "y")
        locked_dices = None
        scored_combination = None

        if score:
            scored_combination = input("Which combination?\n")
        else:
            locked_dices_text = input("Which dices do you want to lock?\n").split()
            locked_dices = [int(x) for x in locked_dices_text]
            print(len(locked_dices))
            while not self.acceptable_locked_dices(locked_dices=locked_dices):
                locked_dices_text = input("Which dices do you want to lock?\n").split()
                locked_dices = [int(x) for x in locked_dices_text]

        return Action(score, locked_dices=locked_dices, scored_combination=scored_combination)

    @staticmethod
    def acceptable_locked_dices(locked_dices):
        if len(locked_dices) > 5 or len(locked_dices) < 0:
            return False

        for dices in locked_dices:
            if dices > 1 or dices < 0:
                return False

        return True
