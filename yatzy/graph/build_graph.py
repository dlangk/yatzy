from yatzy.mechanics import gameengine as Engine
from yatzy.mechanics.gamestate import GameState


def build_graph(combinations_status, upper_section_score):
    """Build the graph of all possible states."""
    widget = Widget(combinations_status, upper_section_score)


class Widget:
    def __init__(self, combinations_status, upper_section_score):
        self.combinations_status = combinations_status
        self.upper_section_score = upper_section_score


class Edge:
    def __init__(self):
        pass


class State:
    def __init__(self, dices, scorecard, rolls):
        pass


state: GameState = Engine.create_initial_state()
playable_combinations = None

build_graph(playable_combinations, 0)
