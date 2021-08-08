import const
import torch
import random
import numpy as np

from action import Action
from engine import Engine
from engine import State
from logger import YatzyLogger

from collections import deque
from yatzy.neural import YatzyNet


class Agent:
    def __init__(self, state_dim, action_dim, save_dir, checkpoint=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=100000)
        self.batch_size = 32

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.gamma = 0.9

        self.curr_step = 0
        self.burnin = 1e3  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync

        self.save_every = 5e5  # no. of experiences between saving YatzyNet Net
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available()

        # YatzyNets's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = YatzyNet(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            # noinspection PyTypeChecker
            self.net = self.net.to(device='cuda')

        if checkpoint:
            self.load(checkpoint)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.logger = YatzyLogger(__name__).get_logger()

    def act_random(self, engine: Engine, state: State):
        self.logger.debug("agent acting")
        reroll = round(random.uniform(0, 1), 2)
        locked_dices = [round(random.uniform(0, 1), 2) for n in range(const.DICES_COUNT)]
        scoring_vector = [round(random.uniform(0, 1), 2) for n in range(const.COMBINATIONS_COUNT)]
        suggested_action = Action(reroll, locked_dices, scoring_vector)
        legal_action = engine.make_action_legal(suggested_action, state)
        self.curr_step += 1
        return legal_action

    def act(self, engine: Engine, state: State):

        # Explore
        if random.uniform(0, 1) < self.exploration_rate:
            self.logger.debug(f"Exploring ({self.exploration_rate})")
            action: Action = self.act_random(engine, state)

        # Exploit
        else:
            self.logger.debug(f"Exploiting ({self.exploration_rate})")
            action: Action = self.act_random(engine, state)

        # Decrease exploration rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        self.curr_step += 1
        return action

    def cache(self, state: State, next_state: State, action: Action, reward, done):
        """
                Store the experience to self.memory (replay buffer)
        """

        state = state.get_tensor()
        next_state = next_state.get_tensor()
        action = action.get_tensor()
        reward = torch.DoubleTensor([reward]).cuda() if self.use_cuda else torch.DoubleTensor([reward])
        done = torch.DoubleTensor([done]).cuda() if self.use_cuda else torch.DoubleTensor([done])

        self.memory.append((state, next_state, action, reward, done,))

    def enough_memory(self, batch_size=None):
        batch_size = batch_size if batch_size else self.batch_size
        return len(self.memory) >= batch_size

    def get_memory(self):
        return self.memory

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_q_online(td_est, td_tgt)

        return td_est.mean().item(), loss

    def td_estimate(self, state, action):
        current_q = self.net(state, model='online')[np.arange(0, self.batch_size), action]  # Q_online(s,a)
        return current_q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_q = self.net(next_state, model='online')
        best_action = torch.argmax(next_state_q, axis=1)
        next_q = self.net(next_state, model='target')[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float()) * self.gamma * next_q).float()

    def update_q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self):
        save_path = self.save_dir / f"yatzy_net_{int(self.curr_step // self.save_every)}.chkpt"

        torch.save(
            dict(
                model=self.net.state_dict(),
                exploration_rate=self.exploration_rate
            ),
            save_path
        )
        print(f"YatzyNet saved to {save_path} at step {self.curr_step}")

    def sync_q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def load(self, load_path):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path, map_location=('cuda' if self.use_cuda else 'cpu'))
        exploration_rate = ckp.get('exploration_rate')
        state_dict = ckp.get('model')

        print(f"Loading model at {load_path} with exploration rate {exploration_rate}")
        self.net.load_state_dict(state_dict)
        self.exploration_rate = exploration_rate
