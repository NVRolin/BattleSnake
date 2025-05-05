import random

def get_action(agent_id: int, obs: dict, board_size: int) -> int:
    my_head = tuple(obs['snakes'][agent_id][0])
    my_body = set(tuple(seg) for seg in obs['snakes'][agent_id][1:])
    my_length = len(obs['snakes'][agent_id])
    food_positions = set(tuple(f) for f in obs['food'])
    other_bodies = set()
    for i, snake in enumerate(obs['snakes']):
        if i != agent_id and obs['alive'][i]:
            ate_food = i in obs.get('food_eaten', [])
            length_other = len(snake)
            for j, seg in enumerate(snake):
                if j == length_other - 1 and not ate_food:
                    continue
                other_bodies.add(tuple(seg))

    possible_moves = {
        0: (my_head[0], my_head[1] - 1),
        1: (my_head[0], my_head[1] + 1),
        2: (my_head[0] - 1, my_head[1]),
        3: (my_head[0] + 1, my_head[1]),
    }

    safe_moves = []
    for action, next_head in possible_moves.items():
        if not (0 <= next_head[0] < board_size and 0 <= next_head[1] < board_size):
            continue
        if next_head in my_body:
            continue
        if next_head in other_bodies:
            potential_opponent_lengths = []
            for i, snake in enumerate(obs['snakes']):
                if i != agent_id and obs['alive'][i] and tuple(snake[0]) == next_head:
                    potential_opponent_lengths.append(len(snake))
            if any(l >= my_length for l in potential_opponent_lengths):
                continue
            if next_head in other_bodies:
                continue
        
        safe_moves.append(action)

    if not safe_moves:
        return 0

    nearest_food = None
    min_dist = float('inf')
    for food in food_positions:
        dist = abs(my_head[0] - food[0]) + abs(my_head[1] - food[1])
        if dist < min_dist:
            min_dist = dist
            nearest_food = food

    food_moves = []
    if nearest_food:
        for action in safe_moves:
            next_head = possible_moves[action]
            dist_to_food = abs(next_head[0] - nearest_food[0]) + abs(next_head[1] - nearest_food[1])
            if dist_to_food < min_dist:
                food_moves.append(action)

    if food_moves:
        return random.choice(food_moves)
    else:
        return random.choice(safe_moves)
