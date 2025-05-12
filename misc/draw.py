from rl.renderer import BattlesnakeRenderer

class DrawBattlesnakeEnvironment:
    def __init__(self, env):
        self.reward = 0
        self.rewards = []
        self.done = False
        self.info = {}
        self.env = env
        self.obs = self.env.reset()
        self.action = []

        # Initialize renderers
        board_size = 11  # Standard Battlesnake board size
        self.renderer = BattlesnakeRenderer(board_size)


    def close(self):
        if hasattr(self, 'renderer') and self.renderer:
            self.renderer.close()

    def reset(self):
        self.reward = 0
        self.rewards = []
        self.done = False
        self.info = {}
        self.obs = self.env.reset()
        self.action = []

    def __update(self, agent):
        if not self.done:
            self.renderer.render(self.env.get_observation(), self.info, fps=10)

            self.env, self.obs, self.action, self.reward, self.done,self.info = agent._forward_processing(
                self.env, self.env.get_observation(), self.action, self.reward, self.done,self.info)
            if self.done:
                self.rewards.append(self.reward)


    def run(self, agent1):
        while not self.done:
            self.__update(agent1)
        return self.rewards
