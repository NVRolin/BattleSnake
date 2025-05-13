# __________         __    __  .__                               __
# \______   \_____ _/  |__/  |_|  |   ____   ______ ____ _____  |  | __ ____
#  |    |  _/\__  \\   __\   __\  | _/ __ \ /  ___//    \\__  \ |  |/ // __ \
#  |    |   \ / __ \|  |  |  | |  |_\  ___/ \___ \|   |  \/ __ \|    <\  ___/
#  |________/(______/__|  |__| |____/\_____>______>___|__(______/__|__\\_____>

import os
import torch
import typing
import numpy as np
from rl.agent import DQNAgent

RL_MODEL = None
MOVE_MAPPING = ["up", "down", "left", "right"]

def load_model(board_size):
    model_path = f"rl/models/candidates/_{board_size}"

    assert os.path.exists(model_path), "No models found in rl/models/candidates"
    
    class tempEnv:
        ACTIONS = [0, 1, 2, 3]
    env = tempEnv()

    return DQNAgent.load_models_and_parameters_DQN_CNN(model_path, env)


def convert_state_to_frames(game_state):
    global prev_food_frame, player_ids

    B = game_state['board']['width']
    health_frame = np.zeros((B, B), dtype=np.uint8)
    bin_body_frame = np.zeros((B, B), dtype=np.uint8)
    segment_body_frame = np.zeros((B, B), dtype=np.uint8)
    longer_opponent_frame = np.zeros((B, B), dtype=np.uint8)
    food_frame = np.zeros((B, B), dtype=np.uint8)
    board_frame = np.full((B, B), 255, dtype=np.uint8)
    agent_head_frame = np.zeros((B, B), dtype=np.uint8)
    double_tail_frame = np.zeros((B, B), dtype=np.uint8) #
    longer_size_frame = np.zeros((B, B), dtype=np.uint8) 
    shorter_size_frame = np.zeros((B, B), dtype=np.uint8)
    alive_count_frames = np.zeros((3, B, B), dtype=np.uint8) # 
    
    my_snake = game_state["you"]

    for snake in game_state["board"]["snakes"]:
        head_x, head_y = snake["head"]["x"], snake["head"]["y"]
        health = snake["health"]
        health_frame[head_y, head_x] = health * 255 // 100
        if snake["id"] == my_snake["id"]:
            agent_head_frame[head_y, head_x] = 255

    for snake in game_state["board"]["snakes"]:

        is_my_snake = snake["id"] == my_snake["id"]
        
        for j, segment in enumerate(snake["body"]):
            x, y = segment["x"], segment["y"]
            bin_body_frame[y, x] = 255
            segment_body_frame[y, x] = j
        
        if not is_my_snake:
            if len(snake["body"]) > len(my_snake["body"]):
                longer_opponent_frame[snake["head"]["y"], snake["head"]["x"]] = 255
                for segment in snake["body"]:
                    x, y = segment["x"], segment["y"]
                    longer_size_frame[y, x] = len(snake["body"]) - len(my_snake["body"])
            elif len(snake["body"]) < len(my_snake["body"]):
                for segment in snake["body"]:
                    x, y = segment["x"], segment["y"]
                    shorter_size_frame[y, x] = len(my_snake["body"]) - len(snake["body"])
    

    eaten_food_positions = np.where((prev_food_frame == 255) & (food_frame == 0))
    if len(eaten_food_positions[0]) > 0:

        for i in range(len(eaten_food_positions[0])):
            eaten_y = eaten_food_positions[0][i]
            eaten_x = eaten_food_positions[1][i]
            
            for snake in game_state["board"]["snakes"]:
                head_x, head_y = snake["head"]["x"], snake["head"]["y"]
                
                if head_x == eaten_x and head_y == eaten_y:
                    tail_x, tail_y = snake["body"][-1]["x"], snake["body"][-1]["y"]
                    double_tail_frame[tail_y, tail_x] = 255
                    break
    
    for food in game_state["board"]["food"]:
        food_frame[food["y"], food["x"]] = 255

    prev_food_frame = food_frame.copy()

    for i in range(len(alive_count_frames)):
        alive_count_frames[i].fill(0)
    
    current_snake_ids = [snake["id"] for snake in game_state["board"]["snakes"]]

    for i, player_id in enumerate(player_ids):
        if player_id in current_snake_ids:
            alive_count_frames[i].fill(255)

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

# info is called when you create your Battlesnake on play.battlesnake.com
def info() -> typing.Dict:
    print("INFO")

    return {
        "apiversion": "1",
        "author": "MD-NR",
        "color": "#3E338F",
        "head": "beluga",
        "tail": "hook",
    }


# start is called when your Battlesnake begins a game
def start(game_state: typing.Dict):
    global RL_MODEL, prev_food_frame, player_ids

    print("GAME START")

    board_width = game_state["board"]["width"]
    board_height = game_state["board"]["height"]
    assert board_width == board_height, "Board is not square"

    prev_food_frame = np.zeros((board_width, board_width), dtype=np.uint8)
    player_ids = [snake["id"] for snake in game_state["board"]["snakes"] if snake["id"] != game_state["you"]["id"]]

    RL_MODEL = load_model(board_width)


# end is called when your Battlesnake finishes a game
def end(game_state: typing.Dict):
    global RL_MODEL
    print("GAME OVER\n")


# move is called on every turn and returns your next move
# Valid moves are "up", "down", "left", or "right"
# See https://docs.battlesnake.com/api/example-move for available data
def move(game_state: typing.Dict) -> typing.Dict:
    global RL_MODEL, MOVE_MAPPING
    
    if RL_MODEL is None:
        board_size = game_state["board"]["width"]
        RL_MODEL = load_model(board_size)
    
    state_tensor = convert_state_to_frames(game_state)
    actions = RL_MODEL._forward_queue(state_tensor)
    
    is_move_safe = {"up": True, "down": True, "left": True, "right": True}
    my_head = game_state["you"]["head"]
    my_body = game_state["you"]["body"]
    board_width = game_state["board"]["width"]
    board_height = game_state["board"]["height"]
    
    # Do not move out of bounds
    if my_head["y"] == board_height - 1:
        is_move_safe["up"] = False
    if my_head["y"] == 0:
        is_move_safe["down"] = False
    if my_head["x"] == 0:
        is_move_safe["left"] = False
    if my_head["x"] == board_width - 1:
        is_move_safe["right"] = False
    
    # Do not backwards into your own neck
    if len(my_body) > 1:
        my_neck = my_body[1]
        if my_neck["x"] < my_head["x"]:
            is_move_safe["left"] = False
        elif my_neck["x"] > my_head["x"]:
            is_move_safe["right"] = False
        elif my_neck["y"] < my_head["y"]:
            is_move_safe["down"] = False
        elif my_neck["y"] > my_head["y"]:
            is_move_safe["up"] = False
    
    # Do not collide with yourself
    for segment in my_body[1:-1]:
        if my_head["x"] == segment["x"] and my_head["y"] + 1 == segment["y"]:
            is_move_safe["up"] = False
        if my_head["x"] == segment["x"] and my_head["y"] - 1 == segment["y"]:
            is_move_safe["down"] = False
        if my_head["x"] - 1 == segment["x"] and my_head["y"] == segment["y"]:
            is_move_safe["left"] = False
        if my_head["x"] + 1 == segment["x"] and my_head["y"] == segment["y"]:
            is_move_safe["right"] = False
    
    # Do not collide with other snakes
    for snake in game_state["board"]["snakes"]:
        if snake["id"] == game_state["you"]["id"]:
            continue
    
        for segment in snake["body"]:
            if my_head["x"] == segment["x"] and my_head["y"] + 1 == segment["y"]:
                is_move_safe["up"] = False
            if my_head["x"] == segment["x"] and my_head["y"] - 1 == segment["y"]:
                is_move_safe["down"] = False
            if my_head["x"] - 1 == segment["x"] and my_head["y"] == segment["y"]:
                is_move_safe["left"] = False
            if my_head["x"] + 1 == segment["x"] and my_head["y"] == segment["y"]:
                is_move_safe["right"] = False
    
    # Avoid head-on with smaller enemies
    deltas = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}
    for snake in game_state["board"]["snakes"]:
        if snake["id"] == game_state["you"]["id"]:
            continue
        if len(snake["body"]) >= len(my_body):
            e_head = snake["head"]
            e_body = snake["body"]
            e_neck = e_body[1] if len(e_body) > 1 else None
            for d, (dx, dy) in deltas.items():
                tx, ty = e_head["x"] + dx, e_head["y"] + dy
                if e_neck and tx == e_neck["x"] and ty == e_neck["y"]:
                    continue
                if not (0 <= tx < board_width and 0 <= ty < board_height):
                    continue
                for m, (mx, my) in deltas.items():
                    if my_head["x"] + mx == tx and my_head["y"] + my == ty:
                        is_move_safe[m] = False

    next_move = None
    for action in actions:
        move_option = MOVE_MAPPING[action]
        if is_move_safe[move_option]:
            next_move = move_option
            break

    if next_move is None:
        next_move = MOVE_MAPPING[actions[0]]
    
    print(f"MOVE {game_state['turn']}: {next_move}")
    return {"move": next_move}


# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})
