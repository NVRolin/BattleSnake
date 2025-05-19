import os
import sys
import torch
import numpy as np
import json
import time
import pygame
from rl.agent import *
from gym.env import BattlesnakeEnv
from gym.renderer import BattlesnakeRenderer


def convert_state_to_frames(obs, board_size,agent, snake_idx=0):
    global prev_food_frame
    B = board_size
    health_frame = np.zeros((B, B), dtype=np.uint8)
    bin_body_frame = np.zeros((B, B), dtype=np.uint8)
    segment_body_frame = np.zeros((B, B), dtype=np.uint8)
    longer_opponent_frame = np.zeros((B, B), dtype=np.uint8)
    food_frame = np.zeros((B, B), dtype=np.uint8)
    board_frame = np.full((B, B), 255, dtype=np.uint8)
    agent_head_frame = np.zeros((B, B), dtype=np.uint8)
    double_tail_frame = np.zeros((B, B), dtype=np.uint8)
    longer_size_frame = np.zeros((B, B), dtype=np.uint8)
    shorter_size_frame = np.zeros((B, B), dtype=np.uint8)
    alive_count_frames = np.zeros((3, B, B), dtype=np.uint8)

    if snake_idx == 0:
        if obs['alive'][1]:
            alive_count_frames[0].fill(255)
        if obs['alive'][2]:
            alive_count_frames[1].fill(255)
        if obs['alive'][3]:
            alive_count_frames[2].fill(255)
    elif snake_idx == 1:
        if obs['alive'][0]:
            alive_count_frames[0].fill(255)
        if obs['alive'][2]:
            alive_count_frames[1].fill(255)
        if obs['alive'][3]:
            alive_count_frames[2].fill(255)
    elif snake_idx == 2:
        if obs['alive'][3]:
            alive_count_frames[0].fill(255)
        if obs['alive'][0]:
            alive_count_frames[1].fill(255)
        if obs['alive'][1]:
            alive_count_frames[2].fill(255)
    elif snake_idx == 3:
        if obs['alive'][2]:
            alive_count_frames[0].fill(255)
        if obs['alive'][0]:
            alive_count_frames[1].fill(255)
        if obs['alive'][1]:
            alive_count_frames[2].fill(255)

    for i, snake in enumerate(obs["snakes"]):
        if not obs["alive"][i]:
            continue
        head_x, head_y = snake[0]
        health = obs["health"][i]
        health_frame[head_y, head_x] = health * 255 // 100
    
    for i, snake in enumerate(obs["snakes"]):
        if not obs["alive"][i]:
            continue

        for j, (x, y) in enumerate(snake):
            bin_body_frame[y, x] = 255
            segment_body_frame[y, x] = j
            
            if i != snake_idx and len(snake) >= len(obs["snakes"][snake_idx]):
                longer_size_frame[y, x] = len(snake) - len(obs["snakes"][snake_idx])
                if j == 0:  # Head
                    longer_opponent_frame[y, x] = 255
            elif i != snake_idx and len(snake) < len(obs["snakes"][snake_idx]):
                shorter_size_frame[y, x] = len(obs["snakes"][snake_idx]) - len(snake)
    
    head_x, head_y = obs["snakes"][snake_idx][0]
    agent_head_frame[head_y, head_x] = 255
    agent.head_positions[snake_idx] = (head_x, head_y)
    eaten_food_positions = np.where((prev_food_frame == 255) & (food_frame == 0))
    for i in range(len(eaten_food_positions[0])):
        eaten_y = eaten_food_positions[0][i]
        eaten_x = eaten_food_positions[1][i]

        for i, snake in enumerate(obs["snakes"]):
            head_x, head_y = snake[0]

            if head_x == eaten_x and head_y == eaten_y:
                tail_x, tail_y = snake[-1]
                double_tail_frame[tail_y, tail_x] = 255
                break
    for x, y in obs["food"]:
        food_frame[y, x] = 255
    prev_food_frame = food_frame.copy()


    all_frames = np.stack([
        health_frame,
        bin_body_frame,
        segment_body_frame,
        longer_opponent_frame,
        food_frame,
        board_frame,
        agent_head_frame,
        double_tail_frame,
        longer_size_frame,
        shorter_size_frame,
        *alive_count_frames
    ], axis=0)

    state_tensor = torch.from_numpy(all_frames)
    state_tensor = state_tensor.unsqueeze(0)
    return state_tensor


def list_available_models():
    models = []
    model_dir = "rl/models/candidates"
    if os.path.exists(model_dir):
        for model_name in os.listdir(model_dir):
            model_path = os.path.join(model_dir, model_name)
            if os.path.isdir(model_path) and "parameters.json" in os.listdir(model_path):
                models.append(model_path)
    return models

def choose_model(models, player):
    print(f"\nAvailable models for Player {player}:")
    for i, model in enumerate(models):
        print(f"{i+1}. {model}")
    choice = input(f"Choose model for Player {player} (1-{len(models)}): ")
    try:
        index = int(choice) - 1
        if 0 <= index < len(models):
            return models[index]
    except ValueError:
        pass
    print("Invalid choice, using default")
    return models[0]

if __name__ == "__main__":
    global prev_food_frame
    models = list_available_models()

    if not models:
        print("No models found in rl/models/candidates directory")
        sys.exit(1)
    
    use_default = input("Use default models? (y/n): ").lower() == 'y'
    
    if use_default and len(models) >= 2:
        model1_path = models[0]
        model2_path = models[0]
        model3_path = models[0]
        model4_path = models[0]
    else:
        model1_path = choose_model(models, 1)
        model2_path = choose_model(models, 1)
        model3_path = choose_model(models, 1)
        model4_path = choose_model(models, 1)
    
    BOARD_SIZE = 11
    N_SNAKES = 4
    env = BattlesnakeEnv(board_size=BOARD_SIZE, n_snakes=N_SNAKES)

    print(f"\nLoading Model 1 from: {model1_path}")
    snake1 = load_dqn_agent(model1_path, env, old_model=False)
    
    print(f"Loading Model 2 from: {model2_path}")
    snake2 = load_dqn_agent(model2_path, env, old_model=False)

    print(f"\nLoading Model 3 from: {model3_path}")
    snake3 = load_dqn_agent(model3_path, env, old_model=False)

    print(f"Loading Model 4 from: {model4_path}")
    snake4 = load_dqn_agent(model4_path, env, old_model=False)
    
    renderer = BattlesnakeRenderer(board_size=BOARD_SIZE, n_snakes=N_SNAKES)
    
    obs = env.reset()
    done = False
    info = {"elimination_causes": [None] * N_SNAKES, "turn": 0}
    running = True
    turn = 0
    rewards_total = [0, 0]
    prev_food_frame = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)
    print("\nStarting match...")
    try:
        while running:
            for event in pygame.event.get():    
                if event.type == pygame.QUIT:
                    running = False
            
            if not done:
                actions = []
                state, rot_state = snake1._stack_frames(env)
                state_tensor = torch.tensor(rot_state, dtype=torch.uint8, requires_grad=False,
                                            device='cuda')
                if obs['alive'][0]:
                    action1 = snake1._forward(state[0], 0, snake1._head_positions[0], state_tensor[0])
                    actions.append(action1)
                else:
                    actions.append(0)
                
                if obs['alive'][1]:
                    action2 = snake2._forward(state[1], 0, snake1._head_positions[1], state_tensor[1])
                    actions.append(action2)
                else:
                    actions.append(0)

                if obs['alive'][2]:
                    action3 = snake3._forward(state[2], 0, snake1._head_positions[2], state_tensor[2])
                    actions.append(action3)
                else:
                    actions.append(0)

                if obs['alive'][3]:
                    action4 = snake4._forward(state[3], 0, snake1._head_positions[3], state_tensor[3])
                    actions.append(action4)
                else:
                    actions.append(0)


                new_obs, rewards, done, info = env.step(actions)
                
                rewards_total[0] += rewards[0]
                rewards_total[1] += rewards[1]
                
                obs = new_obs
                turn += 1
                info["turn"] = turn
                
                renderer.render(obs, info, fps=5)
                
                if done:
                    print("\nMatch Results:")
                    print(f"Turns played: {turn}")
                    print(f"Snake 1 reward: {rewards_total[0]}")
                    print(f"Snake 2 reward: {rewards_total[1]}")
                    print(f"Snake 1 health: {env.snake_health[0]}")
                    print(f"Snake 2 health: {env.snake_health[1]}")
                    print(f"Snake 1 status: {'Alive' if env.snake_alive[0] else 'Dead'}")
                    print(f"Snake 2 status: {'Alive' if env.snake_alive[1] else 'Dead'}")
                    
                    if env.snake_alive[0] and not env.snake_alive[1]:
                        print("Snake 1 wins!")
                    elif not env.snake_alive[0] and env.snake_alive[1]:
                        print("Snake 2 wins!")
                    elif env.snake_alive[0] and env.snake_alive[1]:
                        if rewards_total[0] > rewards_total[1]:
                            print("Snake 1 wins on points!")
                        elif rewards_total[1] > rewards_total[0]:
                            print("Snake 2 wins on points!")
                        else:
                            print("Match is a draw!")
                    else:
                        print("Both snakes died!")
                    
                    renderer.render(obs, info, fps=5)
                    time.sleep(3)
                    
                    play_again = input("\nPlay again? (y/n): ").lower() == 'y'
                    if play_again:
                        obs = env.reset()
                        done = False
                        info = {"elimination_causes": [None] * N_SNAKES, "turn": 0}
                        turn = 0
                        rewards_total = [0, 0]
                    else:
                        running = False
    
    except KeyboardInterrupt:
        print("Match interrupted by user.")