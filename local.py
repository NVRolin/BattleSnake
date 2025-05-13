import os
import sys
import torch
import numpy as np
import json
import time
import pygame
from rl.agent import DQNAgent
from gym.env import BattlesnakeEnv
from gym.renderer import BattlesnakeRenderer

def load_trained_model(model_dir):
    with open(os.path.join(model_dir, "parameters.json"), "r") as f:
        params = json.load(f)
    
    class tempEnv:
        ACTIONS = [0, 1, 2, 3]
    env = tempEnv()

    return DQNAgent.load_models_and_parameters_DQN_CNN(model_dir, env)

def convert_state_to_frames(obs, board_size, snake_idx=0):
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
            
            if i != snake_idx and len(snake) > len(obs["snakes"][snake_idx]):
                longer_size_frame[y, x] = len(snake) - len(obs["snakes"][snake_idx])
                if j == 0:  # Head
                    longer_opponent_frame[y, x] = 255
            elif i != snake_idx and len(snake) < len(obs["snakes"][snake_idx]):
                shorter_size_frame[y, x] = len(obs["snakes"][snake_idx]) - len(snake)
    
    head_x, head_y = obs["snakes"][snake_idx][0]
    agent_head_frame[head_y, head_x] = 255
    
    for x, y in obs["food"]:
        food_frame[y, x] = 255
    
    alive_flags = obs["alive"]
    alive_count = sum(alive_flags)
    other_alive = alive_count - 1
    idx = max(0, min(other_alive-1, 2))
    alive_count_frames[idx, :, :] = 255
    
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
    model_dir = "rl/models/Battlesnake"
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
    models = list_available_models()
    
    if not models:
        print("No models found in rl/models/Battlesnake directory")
        sys.exit(1)
    
    use_default = input("Use default models? (y/n): ").lower() == 'y'
    
    if use_default and len(models) >= 2:
        model1_path = models[0]
        model2_path = models[1]
    else:
        model1_path = choose_model(models, 1)
        model2_path = choose_model(models, 2)
    
    print(f"\nLoading Model 1 from: {model1_path}")
    snake1 = load_trained_model(model1_path)
    
    print(f"Loading Model 2 from: {model2_path}")
    snake2 = load_trained_model(model2_path)
    
    BOARD_SIZE = 11
    N_SNAKES = 2
    
    env = BattlesnakeEnv(board_size=BOARD_SIZE, n_snakes=N_SNAKES)
    renderer = BattlesnakeRenderer(
        board_size=BOARD_SIZE,
        n_snakes=N_SNAKES
    )
    
    obs = env.reset()
    done = False
    info = {"elimination_causes": [None] * N_SNAKES, "turn": 0}
    running = True
    turn = 0
    rewards_total = [0, 0]
    
    print("\nStarting match...")
    try:
        while running:
            for event in pygame.event.get():    
                if event.type == pygame.QUIT:
                    running = False
            
            if not done:
                actions = []
                
                if obs['alive'][0]:
                    state_tensor = convert_state_to_frames(obs, BOARD_SIZE, snake_idx=0)
                    action1 = snake1._forward(state_tensor, 0)
                    actions.append(action1)
                else:
                    actions.append(0)
                
                if obs['alive'][1]:
                    state_tensor = convert_state_to_frames(obs, BOARD_SIZE, snake_idx=1)
                    action2 = snake2._forward(state_tensor, 0)
                    actions.append(action2)
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