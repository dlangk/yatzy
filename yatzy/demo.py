import json

from agent import Agent
from gameenvironment import GameEnvironment
from gameenvironment import GameState
from logger import YatzyLogger

engine: GameEnvironment = GameEnvironment()
state: GameState = engine.get_initial_state()
logger = YatzyLogger(__name__).get_logger()

print(json.dumps(state, indent=4, sort_keys=True))
