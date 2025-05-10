from rl.renderer import BattlesnakeRenderer
class DrawEnvironment:
    def __init__(self, env1, dual, env2=None):
        self.dual = dual
        self.reward1 = self.reward2 = 0
        self.rewards1, self.rewards2 = [], []
        self.done1 = self.done2 = False

        self.env1 = env1
        self.state1 = self.env1.reset()[0]
        self.action1 = 0

        if dual:
            self.env2 = env2
            self.state2 = self.env2.reset()[0]
            self.action2 = 0
        else:
            self.env2 = None

    def close(self):
        self.env1.close()
        if self.dual:
            self.env2.close()

    def reset(self):
        self.reward1 = self.reward2 = 0
        self.rewards1, self.rewards2 = [], []
        self.done1 = self.done2 = False

        self.state1 = self.env1.reset()[0]
        self.action1 = 0

        if self.dual:
            self.state2 = self.env2.reset()[0]
            self.action2 = 0
        else:
            self.state1, *_ = self.env1.step(0)

    def __update(self, agent1, agent2):
        if self.env1.render_mode:
            self.env1.render()
        if self.dual and self.env2.render_mode:
            self.env2.render()

        if not self.done1:
            self.env1, self.state1, self.action1, self.reward1, self.done1 = agent1._forward_processing(
                self.env1, self.state1, self.action1, self.reward1, self.done1)
            if self.done1:
                self.rewards1.append(self.reward1)

        if self.dual and not self.done2:
            self.env2, self.state2, self.action2, self.reward2, self.done2 = agent2._forward_processing(
                self.env2, self.state2, self.action2, self.reward2, self.done2)
            if self.done2:
                self.rewards2.append(self.reward2)

    def run(self, agent1, agent2=None):
        while (self.dual and (not self.done1 or not self.done2)) or not self.done1:
            self.__update(agent1, agent2)
        return self.rewards1, self.rewards2





class DrawBattlesnakeEnvironment:
    def __init__(self, env1, dual, env2=None):
        self.dual = dual
        self.reward1 = self.reward2 = 0
        self.rewards1, self.rewards2 = [], []
        self.done1 = self.done2 = False
        self.info1 = self.info2 = {}
        self.env1 = env1
        self.obs1 = self.env1.reset()
        self.action1 = self.action2 = 0

        # Initialize renderers
        board_size = 11  # Standard Battlesnake board size
        self.renderer1 = BattlesnakeRenderer(board_size)

        if dual:
            self.env2 = env2
            self.obs2 = self.env1.reset()
            self.renderer2 = BattlesnakeRenderer(board_size)
        else:
            self.env2 = None
            self.renderer2 = None

    def close(self):
        if hasattr(self, 'renderer1') and self.renderer1:
            self.renderer1.close()
        if hasattr(self, 'renderer2') and self.renderer2:
            self.renderer2.close()

    def reset(self):
        self.reward1 = self.reward2 = 0
        self.rewards1, self.rewards2 = [], []
        self.done1 = self.done2 = False
        self.info1 = self.info2 = {}
        self.obs1 = self.env1.reset()
        self.action1 = self.action2 = 0

        if self.dual:
            self.obs2 = self.env2.reset()[0]

    def __update(self, agent1, agent2):
        if not self.done1:
            self.renderer1.render(self.env1.get_observation(), self.info1, fps=10)

            self.env1, self.obs1, self.action1, self.reward1, self.done1,self.info1 = agent1._forward_processing(
                self.env1, self.env1.get_observation(), self.action1, self.reward1, self.done1,self.info1)
            if self.done1:
                self.rewards1.append(self.reward1)

        if self.dual and not self.done2:
            # Get observation and info from second environment state
            self.renderer2.render(self.env2.get_observation(), self.info2, fps=10)

            self.env2, self.obs2, self.action2, self.reward2, self.done2,self.info2 = agent2._forward_processing(
                self.env2, self.env2.get_observation(), self.action2, self.reward2, self.done2,self.info2)
            if self.done2:
                self.rewards2.append(self.reward2)

    def run(self, agent1, agent2=None):
        while (self.dual and (not self.done1 or not self.done2)) or not self.done1:
            self.__update(agent1, agent2)
        return self.rewards1, self.rewards2
