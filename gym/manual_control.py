import time
import pygame
from env import BattlesnakeEnv
from renderer import BattlesnakeRenderer

KEY_TO_ACTION = {
    pygame.K_UP: 0,
    pygame.K_DOWN: 1,
    pygame.K_LEFT: 2,
    pygame.K_RIGHT: 3,
}

def main():
    N_SNAKES = 1
    BOARD_SIZE = 11
    env = BattlesnakeEnv(seed=7, board_size=BOARD_SIZE, n_snakes=N_SNAKES)
    renderer = BattlesnakeRenderer(board_size=BOARD_SIZE, n_snakes=N_SNAKES)

    obs = env.reset()
    done = False
    info = {"elimination_causes": [None] * N_SNAKES}
    running = True
    action = 0
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in KEY_TO_ACTION:
                    action = KEY_TO_ACTION[event.key]

        if not done:
            actions = [action]
            new_obs, reward, done, info = env.step(actions)
            obs = new_obs
            renderer.render(obs, info, fps=2)
            clock.tick(10)
            if done:
                renderer.render(obs, info, fps=2)
                time.sleep(2) 
                obs = env.reset()
                info = {"elimination_causes": [None] * N_SNAKES}
                done = False
                action = 0
        else:
            pass

    renderer.close()

if __name__ == "__main__":
    main()
