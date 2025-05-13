import pygame
import numpy as np

class BattlesnakeRenderer:
    def __init__(self, board_size=11, grid_size=500, snake_colors=None, n_snakes=4):
        pygame.init()
        pygame.font.init()

        self.board_size = board_size
        self.side_panel_width = 200
        self.grid_size = grid_size
        self.cell_size = grid_size // board_size

        self.game_width = self.cell_size * board_size
        self.game_height = self.cell_size * board_size
        self.total_width = self.game_width + self.side_panel_width
        self.total_height = max(self.game_height, 100)

        self.screen = pygame.display.set_mode((self.total_width, self.total_height))
        pygame.display.set_caption('Battlesnake')
        self.clock = pygame.time.Clock()

        self.snake_colors = snake_colors or [
             (174, 84, 255),
             (84, 176, 255),
             (255, 199, 84),
             (255, 84, 84)
        ]
        self.food_color = (200, 200, 200)
        self.bg_color = (25, 25, 25)
        self.grid_bg_color = (45, 45, 45)
        self.text_color = (220, 220, 220)
        self.dead_snake_alpha = 100

        self.title_font = pygame.font.SysFont('monospace', 18, bold=True)
        self.info_font = pygame.font.SysFont('monospace', 18, bold=True)
        self.small_info_font = pygame.font.SysFont('monospace', 14, bold=True)
        self.face_font = pygame.font.SysFont('monospace', int(self.cell_size * 0.7), bold=True)

    def _get_snake_face(self, snake_id):
        if snake_id == 0:
            return ":|"
        elif snake_id == 1:
            return ":o"
        elif snake_id == 2:
            return ":("
        else:
            return ":)"

    def render(self, obs, info, fps=10):
        snakes = obs.get('snakes', [])
        food = obs.get('food', [])
        alive = obs.get('alive', [])
        health = obs.get('health', [])
        turn = obs.get('turn', 0)
        elimination_causes = info.get('elimination_causes', [None] * len(snakes))

        num_snakes = len(snakes)
        if len(alive) < num_snakes: alive.extend([False] * (num_snakes - len(alive)))
        if len(health) < num_snakes: health.extend([0] * (num_snakes - len(health)))
        if len(elimination_causes) < num_snakes: elimination_causes.extend([None] * (num_snakes - len(elimination_causes)))

        self.screen.fill(self.bg_color)

        game_surface_rect = pygame.Rect(0, 0, self.game_width, self.game_height)
        game_surface = self.screen.subsurface(game_surface_rect)
        game_surface.fill(self.grid_bg_color)


        for i, snake in enumerate(snakes):
            if not alive[i] and snake:
                color = self.snake_colors[i % len(self.snake_colors)]
                for x, y in snake:
                    temp_surf = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                    temp_surf.set_alpha(int(0.5 * 255))
                    temp_surf.fill((*color, self.dead_snake_alpha))
                    game_surface.blit(temp_surf, (x*self.cell_size, y*self.cell_size))

        for i, snake in enumerate(snakes):
            if alive[i] and snake:
                color = self.snake_colors[i % len(self.snake_colors)]
                for j, (x, y) in enumerate(snake):
                    rect = pygame.Rect(x*self.cell_size, y*self.cell_size, self.cell_size, self.cell_size)
                    pygame.draw.rect(game_surface, color, rect)
                    if j == 0:
                        face_text = self._get_snake_face(i)
                        face_render = self.face_font.render(face_text, True, self.grid_bg_color)
                        face_rect = face_render.get_rect(center=rect.center)
                        game_surface.blit(face_render, face_rect)

        
        for fx, fy in food:
            food_rect = pygame.Rect(fx*self.cell_size, fy*self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(game_surface, self.food_color, food_rect)


        border = 4 
        for x in range(0, self.game_width + 1, self.cell_size):
            pygame.draw.line(game_surface, self.bg_color, (x, 0), (x, self.game_height), border)
        for y in range(0, self.game_height + 1, self.cell_size):
            pygame.draw.line(game_surface, self.bg_color, (0, y), (self.game_width, y), border)

        panel_x_start = self.game_width
        panel_rect = pygame.Rect(panel_x_start, 0, self.side_panel_width, self.total_height)
        panel_surface = self.screen.subsurface(panel_rect)
        panel_surface.fill(self.bg_color)

        turn_text = self.title_font.render(f"TURN {turn}", True, self.text_color)
        panel_surface.blit(turn_text, (self.side_panel_width // 2 - turn_text.get_width() // 2, 20))

        y_offset = 60
        bar_height = 15
        bar_max_width = self.side_panel_width * 0.65
        spacing = 65

        for i in range(num_snakes):
            agent_name = f"Agent {i}"
            name_text = self.info_font.render(agent_name, True, self.snake_colors[i % len(self.snake_colors)])
            panel_surface.blit(name_text, (20, y_offset))

            if alive[i]:
                bg_bar_rect = pygame.Rect(20, y_offset + 25, bar_max_width, bar_height)
                pygame.draw.rect(panel_surface, (60,60,60), bg_bar_rect, border_radius=3)

                health_ratio = max(0, min(1, health[i]/100))
                bar_width = int(bar_max_width * health_ratio)
                fg_bar_rect = pygame.Rect(20, y_offset + 25, bar_width, bar_height)
                if bar_width > 0:
                    pygame.draw.rect(panel_surface, self.snake_colors[i % len(self.snake_colors)], fg_bar_rect, border_radius=3)

                health_val_text = self.small_info_font.render(str(health[i]), True, self.text_color)
                panel_surface.blit(health_val_text, (20 + bar_max_width + 10, y_offset + 23))
            else:
                cause = elimination_causes[i] if elimination_causes[i] else "Eliminated"
                if not isinstance(cause, str):
                    cause = str(cause)
                max_text_width = self.side_panel_width - 40
                words = cause.split(' ')
                lines = []
                current_line = ""
                for word in words:
                    test_line = current_line + word + " "
                    test_render = self.small_info_font.render(test_line, True, (150, 150, 150))
                    if test_render.get_width() <= max_text_width:
                        current_line = test_line
                    else:
                        lines.append(current_line.strip())
                        current_line = word + " "
                lines.append(current_line.strip())

                line_y = y_offset + 25
                for line in lines:
                    elim_text = self.small_info_font.render(line, True, (150, 150, 150))
                    panel_surface.blit(elim_text, (20, line_y))
                    line_y += self.small_info_font.get_height() + 2

            y_offset += spacing

        pygame.display.flip()
        self.clock.tick(fps)

    def close(self):
        pygame.font.quit()
        pygame.quit()
