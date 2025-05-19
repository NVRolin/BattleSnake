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

def load_model():
    model_path = f"rl/models/candidates/_00"

    assert os.path.exists(model_path), "No model found in rl/models/candidates"
    
    class tempEnv:
        ACTIONS = [0, 1, 2, 3]
    env = tempEnv()

    return DQNAgent.load_dqn_agent(model_path, env)


def convert_state_to_frames(game_state,RL_MODEL):
    global prev_food_frame, player_ids
    B = game_state['board']['width']
    health_frame = np.zeros((B, B), dtype=np.uint8)  # health at head
    bin_body_frame = np.zeros((B, B), dtype=np.uint8)  # snake body with values 255
    segment_body_frame = np.zeros((B, B), dtype=np.uint8)  # snake body with increasing segment length
    longer_opponent_frame = np.zeros((B, B), dtype=np.uint8)  # longer opponent head with value 255
    food_frame = np.zeros((B, B), dtype=np.uint8)  # food positions with value 255
    board_frame = np.full((B, B), 255, dtype=np.uint8)  # board with value 255
    agent_head_frame = np.zeros((B, B), dtype=np.uint8)  # agent heads with value 255
    double_tail_frame = np.zeros((B, B), dtype=np.uint8)  # double tail with value 255
    longer_size_frame = np.zeros((B, B), dtype=np.uint8)  # longer opponent snake body with value 255
    shorter_size_frame = np.zeros((B, B), dtype=np.uint8)  # shorter opponent snake body with value 255
    alive_count_frames = np.zeros((3, B, B), dtype=np.uint8)  # alive count frames with value 255




    my_snake = game_state["you"]
    i = 1
    for snake in game_state["board"]["snakes"]:
        head_x, head_y = snake["head"]["x"], snake["head"]["y"]
        health = snake["health"]
        health_frame[head_y, head_x] = health * 255 // 100
        if snake["id"] == my_snake["id"]:
            agent_head_frame[head_y, head_x] = 255
            RL_MODEL._head_positions[0] = (head_x, head_y)
        elif snake["name"] == my_snake["name"]:
            alive_count_frames[0].fill(255)
            RL_MODEL._head_positions[1] = (head_x, head_y)
        else:
            alive_count_frames[i].fill(255)
            RL_MODEL._head_positions[i] = (head_x, head_y)
            i += 1

        for j, segment in enumerate(snake["body"]):
            x, y = segment["x"], segment["y"]
            bin_body_frame[y, x] = 255
            segment_body_frame[y, x] = j
            if len(snake["body"]) >= len(my_snake["body"]):
                longer_size_frame[y, x] = len(snake["body"]) - len(my_snake["body"])
            else:
                shorter_size_frame[y, x] = len(my_snake["body"]) - len(snake["body"])
        if len(snake["body"]) >= len(my_snake["body"]):
            longer_opponent_frame[snake["head"]["y"], snake["head"]["x"]] = 255

    # Check for double tail
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

    
    return all_frames

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

    RL_MODEL = load_model()


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
        RL_MODEL = load_model()
    
    state = convert_state_to_frames(game_state,RL_MODEL)
    state_tensor = torch.tensor(state, dtype=torch.uint8, requires_grad=False)
    action = RL_MODEL._forward(state_tensor,0,RL_MODEL._head_positions[0])


    print(f"MOVE {game_state['turn']}: {action}")
    return {"move": action}


# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})
