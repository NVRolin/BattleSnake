import numpy as np
import random
from typing import List, Tuple, Dict, Optional

class BattlesnakeEnv:
    DIRECTIONS = {
        0: (0, -1),
        1: (0, 1),
        2: (-1, 0),
        3: (1, 0),
    }
    ACTIONS = [0, 1, 2, 3]
    ACTION_NAMES = ['up', 'down', 'left', 'right']
    NAME = 'Battlesnake'
    def __init__(self, board_size: int = 11, n_snakes: int = 1, seed: Optional[int] = None):
        self.food_eaten = [False] * n_snakes
        self.board_size = board_size
        self.n_snakes = max(1, min(4, n_snakes))
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        self.board = np.zeros((board_size, board_size), dtype=np.int32)
        self.snakes: List[List[Tuple[int, int]]] = []
        self.snake_alive: List[bool] = []
        self.snake_health: List[int] = []
        self.elimination_causes: List[Optional[str]] = []
        self.food: List[Tuple[int, int]] = []
        self.turns = 0
        self.starting_length = 3
        self.target_lengths: List[int] = []

    def reset(self) -> Dict:
        self.board.fill(0)
        self.snakes = []
        self.snake_alive = []
        self.snake_health = []
        self.elimination_causes = []
        self.food = []
        self.turns = 0
        self.target_lengths = []

        placements1 = [
            (self.board_size // 2, 1),
            (self.board_size // 2, self.board_size - 2),
            (1, self.board_size // 2),
            (self.board_size - 2, self.board_size // 2),
        ]
        placements2 = [
            (1, 1),
            (self.board_size - 2, self.board_size - 2),
            (1, self.board_size - 2),
            (self.board_size - 2, 1),
        ]
        if random.random() < 0.5:
            placements = placements1
        else:
            placements = placements2
        random.shuffle(placements)
        for i in range(self.n_snakes):
            # Shuffle the placements
            x, y = placements[i]
            self.snakes.append([(x, y)])
            self.board[y, x] = 2 + i
            self.snake_alive.append(True)
            self.snake_health.append(100)
            self.elimination_causes.append(None)
            self.target_lengths.append(self.starting_length)

        self.spawn_food()
        return self.get_observation()

    def step(self, actions: List[int]) -> Tuple[Dict, List[float], bool, Dict]:
        if len(actions) != self.n_snakes:
            raise ValueError(f"Expected {self.n_snakes} actions, got {len(actions)}")
        self.turns += 1
        rewards = [0.0] * self.n_snakes
        done = False

        next_heads = []
        moved_snakes = [False] * self.n_snakes

        tails_moving_away = set()
        for i in range(self.n_snakes):
            if not self.snake_alive[i]:
                continue
            snake = self.snakes[i]
            if len(snake) > 1:
                growing = (len(snake) < self.target_lengths[i])
                if not growing:
                    tails_moving_away.add(snake[-1])

        for i in range(self.n_snakes):
            if not self.snake_alive[i]:
                next_heads.append(None)
                continue
            snake = self.snakes[i]
            head_x, head_y = snake[0]
            dx, dy = self.DIRECTIONS[actions[i]]
            new_head = (head_x + dx, head_y + dy)
            if len(snake) > 1 and snake[1] == new_head:
                dx, dy = self.DIRECTIONS[(actions[i] + 2) % 4]
                new_head = (head_x + dx, head_y + dy)
            next_heads.append(new_head)
            moved_snakes[i] = True

        for i in range(self.n_snakes):
            if not self.snake_alive[i] or next_heads[i] is None:
                continue
            new_head = next_heads[i]
            if not (0 <= new_head[0] < self.board_size and 0 <= new_head[1] < self.board_size):
                self.snake_alive[i] = False
                self.elimination_causes[i] = "wall_collision"
                rewards[i] = -1.0
                next_heads[i] = None
                for x, y in self.snakes[i]:
                    if 0 <= x < self.board_size and 0 <= y < self.board_size:
                        self.board[y, x] = 0

        head_positions = {}
        for i in range(self.n_snakes):
            if not self.snake_alive[i] or next_heads[i] is None:
                continue
            x, y = next_heads[i]
            if (x, y) in head_positions:
                head_positions[(x, y)].append(i)
            else:
                head_positions[(x, y)] = [i]
            if 0 <= x < self.board_size and 0 <= y < self.board_size and self.board[y, x] >= 2:
                collided_snake_id = self.board[y, x] - 2
                if (x, y) in tails_moving_away:
                    pass
                elif collided_snake_id == i:
                    self.snake_alive[i] = False
                    self.elimination_causes[i] = "self_collision"
                    rewards[i] = -1.0
                    next_heads[i] = None
                    for sx, sy in self.snakes[i]:
                        if 0 <= sx < self.board_size and 0 <= sy < self.board_size:
                            self.board[sy, sx] = 0
                else:
                    self.snake_alive[i] = False
                    self.elimination_causes[i] = "body_collision"
                    rewards[i] = -1.0
                    next_heads[i] = None
                    for sx, sy in self.snakes[i]:
                        if 0 <= sx < self.board_size and 0 <= sy < self.board_size:
                            self.board[sy, sx] = 0

        for pos, snake_ids in head_positions.items():
            if len(snake_ids) <= 1:
                continue
            lengths = [(len(self.snakes[i]), i) for i in snake_ids if self.snake_alive[i]]
            if not lengths:
                continue
            lengths.sort(reverse=True)
            longest_length = lengths[0][0]
            winners = [l[1] for l in lengths if l[0] == longest_length]
            if len(winners) == len(snake_ids):
                for i in snake_ids:
                    if self.snake_alive[i]:
                        self.snake_alive[i] = False
                        self.elimination_causes[i] = "head_to_head"
                        rewards[i] = -1.0
                        next_heads[i] = None
                        for x, y in self.snakes[i]:
                            if 0 <= x < self.board_size and 0 <= y < self.board_size:
                                self.board[y, x] = 0
            else:
                for i in snake_ids:
                    if i not in winners and self.snake_alive[i]:
                        self.snake_alive[i] = False
                        self.elimination_causes[i] = "head_to_head"
                        rewards[i] = -1.0
                        next_heads[i] = None
                        for x, y in self.snakes[i]:
                            if 0 <= x < self.board_size and 0 <= y < self.board_size:
                                self.board[y, x] = 0

        self.food_eaten = [False] * self.n_snakes
        for i in range(self.n_snakes):
            if not self.snake_alive[i] or next_heads[i] is None:
                continue
            x, y = next_heads[i]
            if 0 <= x < self.board_size and 0 <= y < self.board_size and self.board[y, x] == 1:
                self.food_eaten[i] = True
                self.snake_health[i] = 100
                rewards[i] = 1
                self.food.remove((x, y))
                self.board[y, x] = 0
                self.target_lengths[i] += 1

        for i in range(self.n_snakes):
            if not self.snake_alive[i] or next_heads[i] is None:
                continue
            snake = self.snakes[i]
            x, y = next_heads[i]
            snake.insert(0, (x, y))
            if 0 <= x < self.board_size and 0 <= y < self.board_size:
                self.board[y, x] = 2 + i
            current_length = len(snake)
            target_length = self.target_lengths[i]
            if current_length > target_length:
                tail_x, tail_y = snake.pop()
                if 0 <= tail_x < self.board_size and 0 <= tail_y < self.board_size:
                    self.board[tail_y, tail_x] = 0

        for i in range(self.n_snakes):
            if self.snake_alive[i] and not self.food_eaten[i]:
                self.snake_health[i] -= 1
                if self.snake_health[i] <= 0:
                    self.snake_alive[i] = False
                    self.elimination_causes[i] = "starvation"
                    rewards[i] = -1.0
                    for x, y in self.snakes[i]:
                        if 0 <= x < self.board_size and 0 <= y < self.board_size:
                            self.board[y, x] = 0

        self.spawn_food()

        #done = all(not alive for alive in self.snake_alive)
        # When we train we dont want to continue when our snake dies
        done = False
        if not self.snake_alive[0]:
            done = True
        # If we win we are also done
        if not self.snake_alive[2] and not self.snake_alive[3]:
            done = True
            rewards[0] = 1


        info = {
            "turn": self.turns,
            "elimination_causes": self.elimination_causes.copy()
        }

        return self.get_observation(), rewards, done, info

    def spawn_food(self):
        needed = max(1, sum(1 for alive in self.snake_alive if alive)) - len(self.food)
        while needed > 0:
            empty = list(zip(*np.where(self.board == 0)))
            if not empty:
                break
            pos = random.choice(empty)
            self.board[pos[0], pos[1]] = 1
            self.food.append((pos[1], pos[0]))
            needed -= 1

    def get_observation(self) -> Dict:
        return {
            "board": self.board.copy(),
            "snakes": [s.copy() for s in self.snakes],
            "food": self.food.copy(),
            "alive": self.snake_alive.copy(),
            "health": self.snake_health.copy(),
            "food_eaten": self.food_eaten.copy(),
            "turn": self.turns,
            "elimination_causes": self.elimination_causes.copy()
        }