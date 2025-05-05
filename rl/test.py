import time
import pygame
from env import BattlesnakeEnv
from renderer import BattlesnakeRenderer
from simple_agent import get_action

if __name__ == "__main__":
    N_SNAKES = 4
    BOARD_SIZE = 7

    env = BattlesnakeEnv(seed=10, board_size=BOARD_SIZE, n_snakes=N_SNAKES)
    renderer = BattlesnakeRenderer(
        board_size=BOARD_SIZE,
        n_snakes=N_SNAKES
    )

    obs = env.reset()
    done = False
    info = {"elimination_causes": [None] * N_SNAKES}
    running = True
    while running:

        for event in pygame.event.get():    
            if event.type == pygame.QUIT:
                running = False

        if not done:
            actions = []
            for i in range(env.n_snakes):
                if obs['alive'][i]:
                    action = get_action(i, obs, env.board_size)
                else:
                    action = 0
                actions.append(action)

            new_obs, reward, done, info = env.step(actions)
            obs = new_obs

            renderer.render(obs, info, fps=5)

            if done:
                renderer.render(obs, info, fps=5)
                time.sleep(3)
                obs = env.reset()
                info = {"elimination_causes": [None] * N_SNAKES}
                done = False
        else:
            pass

    renderer.close()