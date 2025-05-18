from collections import deque, namedtuple
import random

import numpy as np
import torch
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ExperienceReplayBuffer(object):
    """ Class used to store a buffer containing experiences of the RL agent.
    """

    def __init__(self, capacity,alpha=0.6, beta_start=0.4,beta_frames=100000):
        # Create buffer of maximum length using deque function (check documentation)
        self.__capacity = capacity
        self.buffer = []  # list of experiences
        self.priorities = []  # list of priorities
        self.pos = 0
        self.alpha = alpha  # how much prioritization is used (0 = uniform, 1 = full)
        # importance-sampling weight hyperparameters
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

    def _get_priority(self, error, epsilon=1e-6):
        prio = (abs(error) + epsilon) ** self.alpha
        if np.isnan(prio) or np.isinf(prio) or prio <= 0:
            return epsilon ** self.alpha
        return prio

    def append(self, experience,error=None):
        # Append experience to the buffer
        if error is None and self.priorities:
            priority = max(self.priorities)
        else:
            priority = self._get_priority(error if error is not None else 1.0)

        if len(self.buffer) < self.__capacity:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            # overwrite oldest
            self.buffer[self.pos] = experience
            self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.__capacity

    def get_capacity(self):
        return self.__capacity

    def sample_batch(self, batch_size):
        """ Function used to sample experiences from the buffer.
            returns 5 lists, each of size n. Returns a list of state, actions,
            rewards, next states and done variables.
        """
        # If we try to sample more elements that what are available from the buffer we raise an error
        if batch_size > len(self.buffer):
            raise MemoryError('Requested more data than what is available from the buffer!')
        # compute sampling probabilities
        scaled_prios = np.array(self.priorities) ** self.alpha

        # Check for NaNs, Infs, or all-zero
        if (
                np.any(np.isnan(scaled_prios)) or
                np.any(np.isinf(scaled_prios)) or
                scaled_prios.sum() == 0
        ):
            sample_probs = np.ones_like(scaled_prios) / len(scaled_prios)
        else:
            sample_probs = scaled_prios / scaled_prios.sum()

        # sample indices according to probabilities

        indices = np.random.choice(len(self.buffer), batch_size, p=sample_probs)

        # linearly anneal beta from beta_start to 1
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)
        self.frame += 1

        # compute IS weights
        weights = (len(self.buffer) * sample_probs[indices]) ** (-beta)
        weights /= weights.max()  # normalize for stability
        weights = torch.tensor(weights, dtype=torch.float32,
                               device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # collect experiences
        batch = [self.buffer[idx] for idx in indices]
        batch = Experience(*zip(*batch))

        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        next_states = torch.cat(batch.next_state)
        dones = torch.cat(batch.done)

        return states, actions, rewards, next_states, dones,indices,weights

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = self._get_priority(error)

    def __len__(self):
        return len(self.buffer)