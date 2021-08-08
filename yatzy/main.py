from action import Action
from agent import Agent
from engine import Engine
from engine import State
from logger import YatzyLogger

import datetime
from pathlib import Path

save_dir = Path('./checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

engine = Engine()
agent = Agent(state_dim=41, action_dim=27, save_dir=save_dir)

logger = YatzyLogger(__name__).get_logger()


def main():
    episodes = 10000

    for turn in range(episodes):
        state: State = engine.get_initial_state()
        logger.info(f"starting episode {turn}/{episodes} - agent steps = {agent.curr_step}")

        while not Engine.game_over(state):
            logger.debug(f"state = {state}")
            action: Action = agent.act(engine, state)
            next_state, reward, game_over = engine.step(state, action)

            agent.cache(state, next_state, action, reward, game_over)
            agent.learn()

            logger.debug(f"scorecard = {next_state.get_simple_scorecard()}")

        logger.debug(f"game over!")
        logger.debug(f"final score = {state.get_score()}")


if __name__ == "__main__":
    main()
