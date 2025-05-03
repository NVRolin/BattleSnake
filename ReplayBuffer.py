from collections import deque, namedtuple
import random
import torch
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ExperienceReplayBuffer(object):
    """ Class used to store a buffer containing experiences of the RL agent.
    """

    def __init__(self, capacity):
        # Create buffer of maximum length using deque function (check documentation)
        self.__capacity = capacity
        self.__buffer = deque(maxlen=self.__capacity)

    def append(self, experience):
        # Append experience to the buffer
        self.__buffer.append(experience)

    def get_capacity(self):
        return self.__capacity

    def sample_batch(self, batch_size):
        """ Function used to sample experiences from the buffer.
            returns 5 lists, each of size n. Returns a list of state, actions,
            rewards, next states and done variables.
        """
        # If we try to sample more elements that what are available from the buffer we raise an error
        if batch_size > len(self.__buffer):
            raise MemoryError('Requested more data than what is available from the buffer!')

        # Sample without replacement the indices of the experiences
        # random.sample samples self.buffer batch_size nr of times
        indices = random.sample(self.__buffer, batch_size)
        # Using the indices that we just sampled we construct the batch as separate tuples of the chosen experiences
        batch = Experience(*zip(*indices))

        # Concatenate the tensors in the tuple batch
        # [[1, 2],
        # [3, 4]]
        # concatenated at dim 0
        # [[5, 6],
        # [7, 8]]
        # is
        # [[1, 2],
        # [3, 4],
        # [5, 6],
        # [7, 8]]
        state, action, reward, next_state, done = map(torch.cat, [*batch])

        return state, action, reward, next_state, done

    def clear_memory(self):
        self.__buffer.clear()

