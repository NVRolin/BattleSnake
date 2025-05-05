class DrawEnvironment():
    def __init__(self, env1, dual, env2=None):
        self.__dual = dual
        self.__reward1 = 0
        self.__reward2 = 0
        self.__rewards1 = []
        self.__rewards2 = []
        self.__done1 = False
        self.__done2 = False

        if self.__dual is True:
            self.__draw_env1 = env1
            self.__draw_state1 = self.__draw_env1.reset()[0]
            self.__draw_action1 = 0
            self.__draw_env2 = env2
            self.__draw_state2 = self.__draw_env2.reset()[0]
            self.__draw_action2 = 0
        else:
            self.__draw_env1 = env1
            self.__draw_state1 = self.__draw_env1.reset()[0]

            self.__draw_action1 = 0

            self.__draw_env2 = None

    def close(self):
        if self.__dual is True:
            self.__draw_env1 = self.__draw_env1.close()
            self.__draw_env2 = self.__draw_env2.close()
        else:
            self.__draw_env1 = self.__draw_env1.close()

    def reset(self):
        self.__reward1 = 0
        self.__reward2 = 0
        self.__rewards1 = []
        self.__rewards2 = []
        self.__done1 = False
        self.__done2 = False
        if self.__dual is True:
            self.__draw_env1.reset()
            self.__draw_state1 = self.__draw_env1.reset()[0]
            self.__draw_action1 = 0
            self.__draw_env2.reset()
            self.__draw_state2 = self.__draw_env2.reset()[0]
            self.__draw_action2 = 0
        else:
            self.__draw_env1.reset()
            self.__draw_state1 = self.__draw_env1.reset()[0]
            self.__draw_state1, _, _, _, _ = self.__draw_env1.step(0)
            self.__draw_action1 = 0
            self.__draw_env2 = None

    def __update(self, agent1, agent2):
        if self.__dual is True:
            if self.__draw_env1.render_mode is not None:
                self.__draw_env1.render()
            if self.__draw_env2.render_mode is not None:
                self.__draw_env2.render()

            if self.__done1 is False:
                self.__draw_env1, self.__draw_state1, self.__draw_action1, self.__reward1, self.__done1 = agent1._forward_processing(
                    self.__draw_env1, self.__draw_state1, self.__draw_action1, self.__reward1, self.__done1)
                if self.__done1 is True:
                    self.__rewards1.append(self.__reward1)

            if self.__done2 is False:
                self.__draw_env2, self.__draw_state2, self.__draw_action2, self.__reward2, self.__done2 = agent2._forward_processing(
                    self.__draw_env2, self.__draw_state2, self.__draw_action2, self.__reward2, self.__done2)
                if self.__done2 is True:
                    self.__rewards2.append(self.__reward2)
        else:
            if self.__draw_env1.render_mode is not None:
                self.__draw_env1.render()
            if self.__done1 is False:
                self.__draw_env1, self.__draw_state1, self.__draw_action1, self.__reward1, self.__done1 = agent1._forward_processing(
                    self.__draw_env1, self.__draw_state1, self.__draw_action1, self.__reward1, self.__done1)
                if self.__done1 is True:
                    self.__rewards1.append(self.__reward1)

    def run(self, agent1, agent2=None):
        # pyglet.clock.schedule_interval(lambda dt: self.__update(agent1, agent2), 1.0 / 60.0)
        while ((self.__done1 is not True or self.__done2 is not True) and self.__dual) or self.__done1 is not True:
            self.__update(agent1, agent2)
        return self.__rewards1, self.__rewards2