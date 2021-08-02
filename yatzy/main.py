from agent import Agent
from action import Action
from engine import Engine
from engine import State
from logger import YatzyLogger

engine = Engine()
agent = Agent()

logger = YatzyLogger(__name__).get_logger()


def main():
    episodes = 1
    for turn in range(episodes):
        state: State = engine.get_initial_state()
        logger.info(f"starting episode {episodes}")
        while not Engine.game_over(state):
            logger.debug(f"state = {state}")
            action: Action = agent.act(engine, state)
            logger.debug(f"action = {action}")
            new_state, reward, game_over = engine.step(state, action)
            logger.debug(f"scorecard = {new_state.get_simple_scorecard()}")
        logger.info(f"game over!")
        logger.info(f"final score = {state.get_score()}")


if __name__ == "__main__":
    main()
