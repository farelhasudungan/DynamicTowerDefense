import pygame
import math
import heapq
from dataclasses import dataclass
from typing import List, Tuple, Optional


# Initialize Pygame
pygame.init()
pygame.mixer.init()


# Constants
WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080
GRID_SIZE = 35

# UI Layout constants
TOP_BAR_HEIGHT = 100
BOTTOM_BAR_HEIGHT = 100
GAME_AREA_HEIGHT = WINDOW_HEIGHT - TOP_BAR_HEIGHT - BOTTOM_BAR_HEIGHT
GAME_AREA_Y_OFFSET = TOP_BAR_HEIGHT

GRID_WIDTH = WINDOW_WIDTH // GRID_SIZE
GRID_HEIGHT = GAME_AREA_HEIGHT // GRID_SIZE
FPS = 60


# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (34, 139, 34)
RED = (220, 20, 60)
BLUE = (30, 144, 255)
YELLOW = (255, 215, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
DARK_GREEN = (0, 100, 0)
PURPLE = (147, 112, 219)
UI_BACKGROUND = (240, 240, 240)


# Game States
STATE_MENU = "menu"
STATE_OPTIONS = "options"
STATE_GAME = "game"

# Algorithm types
ALGORITHM_ASTAR = "A*"
ALGORITHM_GBFS = "GBFS"
ALGORITHM_DIJKSTRA = "Dijkstra"


@dataclass
class Node:
    x: int
    y: int
    g: float = float('inf')
    h: float = 0
    f: float = float('inf')
    parent: Optional['Node'] = None
    
    def __lt__(self, other):
        return self.f < other.f
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))


class Button:
    """Image-based button for menu and options"""
    def __init__(self, x: int, y: int, normal_image_path: str, hover_image_path: str, action):
        self.x = x
        self.y = y
        self.action = action
        
        try:
            self.normal_image = pygame.image.load(normal_image_path)
            self.hover_image = pygame.image.load(hover_image_path)
        except: # fallback to colored rectangles if images not found
            self.normal_image = pygame.Surface((300, 80))
            self.normal_image.fill(BLUE)
            self.hover_image = pygame.Surface((300, 80))
            self.hover_image.fill(PURPLE)
        
        self.width = self.normal_image.get_width()
        self.height = self.normal_image.get_height()
        self.rect = pygame.Rect(x, y, self.width, self.height)
        self.hovered = False
    
    def update(self, mouse_pos):
        self.hovered = self.rect.collidepoint(mouse_pos)
    
    def draw(self, screen):
        image = self.hover_image if self.hovered else self.normal_image
        screen.blit(image, (self.x, self.y))
    
    def is_clicked(self, mouse_pos):
        return self.rect.collidepoint(mouse_pos)


class UIButton:
    """Simple text-based button for in-game UI"""
    def __init__(self, x: int, y: int, width: int, height: int, text: str, action, font=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.action = action
        self.hovered = False
        self.selected = False
        self.font = font if font else pygame.font.Font(None, 32)
    
    def update(self, mouse_pos):
        self.hovered = self.rect.collidepoint(mouse_pos)
    
    def draw(self, screen):
        if self.selected:
            color = GREEN
        elif self.hovered:
            color = PURPLE
        else:
            color = BLUE
            
        pygame.draw.rect(screen, color, self.rect, border_radius=8)
        pygame.draw.rect(screen, BLACK, self.rect, 2, border_radius=8)
        
        text_surface = self.font.render(self.text, True, WHITE)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)
    
    def is_clicked(self, mouse_pos):
        return self.rect.collidepoint(mouse_pos)


class PathfindingAlgorithms:
    """Collection of pathfinding algorithms with debug output"""
    
    @staticmethod
    def heuristic(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Manhattan distance heuristic"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    @staticmethod
    def get_neighbors(node: Node, grid_width: int, grid_height: int) -> List[Tuple[int, int]]:
        """Get valid neighboring grid positions"""
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        for dx, dy in directions:
            nx, ny = node.x + dx, node.y + dy
            if 0 <= nx < grid_width and 0 <= ny < grid_height:
                neighbors.append((nx, ny))
        
        return neighbors
    
    @staticmethod
    def astar(start: Tuple[int, int], goal: Tuple[int, int], 
              obstacles: set, grid_width: int, grid_height: int, debug: bool = True) -> Optional[List[Tuple[int, int]]]:
        """A* Algorithm: Uses both g(n) and h(n) -> f(n) = g(n) + h(n)"""
        if debug:
            print("\n" + "="*60)
            print("🔍 A* PATHFINDING DEBUG")
            print("="*60)
            print(f"Start: {start} → Goal: {goal}")
            print(f"Obstacles: {len(obstacles)} towers placed")
        
        start_node = Node(start[0], start[1], g=0)
        start_node.h = PathfindingAlgorithms.heuristic(start, goal)
        start_node.f = start_node.h
        
        open_set = []
        heapq.heappush(open_set, start_node)
        closed_set = set()
        node_dict = {(start[0], start[1]): start_node}
        
        nodes_explored = 0
        
        while open_set:
            current = heapq.heappop(open_set)
            nodes_explored += 1
            
            if debug and nodes_explored % 50 == 0:
                print(f"Nodes explored: {nodes_explored} | Open set: {len(open_set)} | Closed set: {len(closed_set)}")
            
            if (current.x, current.y) == goal:
                path = []
                while current:
                    path.append((current.x, current.y))
                    current = current.parent
                path = path[::-1]
                
                if debug:
                    print(f"\n✅ PATH FOUND!")
                    print(f"Total nodes explored: {nodes_explored}")
                    print(f"Path length: {len(path)} steps")
                    print(f"Path cost (g): {len(path) - 1}")
                    print(f"First 10 steps: {path[:10]}")
                    print("="*60)
                
                return path
            
            closed_set.add((current.x, current.y))
            
            for nx, ny in PathfindingAlgorithms.get_neighbors(current, grid_width, grid_height):
                if (nx, ny) in closed_set or (nx, ny) in obstacles:
                    continue
                
                tentative_g = current.g + 1
                
                if (nx, ny) not in node_dict:
                    neighbor = Node(nx, ny)
                    node_dict[(nx, ny)] = neighbor
                else:
                    neighbor = node_dict[(nx, ny)]
                
                if tentative_g < neighbor.g:
                    neighbor.parent = current
                    neighbor.g = tentative_g
                    neighbor.h = PathfindingAlgorithms.heuristic((nx, ny), goal)
                    neighbor.f = neighbor.g + neighbor.h
                    
                    if neighbor not in open_set:
                        heapq.heappush(open_set, neighbor)
        
        if debug:
            print(f"\n❌ NO PATH FOUND!")
            print(f"Total nodes explored: {nodes_explored}")
            print("="*60)
        
        return None
    
    @staticmethod
    def greedy_best_first(start: Tuple[int, int], goal: Tuple[int, int], 
                          obstacles: set, grid_width: int, grid_height: int, debug: bool = True) -> Optional[List[Tuple[int, int]]]:
        """Greedy Best-First Search: Uses only h(n) heuristic"""
        if debug:
            print("\n" + "="*60)
            print("🎯 GREEDY BEST-FIRST SEARCH DEBUG")
            print("="*60)
            print(f"Start: {start} → Goal: {goal}")
            print(f"Obstacles: {len(obstacles)} towers placed")
            print("Strategy: Only using heuristic h(n) - greedy approach")
        
        start_node = Node(start[0], start[1], g=0)
        start_node.h = PathfindingAlgorithms.heuristic(start, goal)
        start_node.f = start_node.h
        
        open_set = []
        heapq.heappush(open_set, start_node)
        closed_set = set()
        node_dict = {(start[0], start[1]): start_node}
        
        nodes_explored = 0
        
        while open_set:
            current = heapq.heappop(open_set)
            nodes_explored += 1
            
            if debug and nodes_explored % 50 == 0:
                print(f"Nodes explored: {nodes_explored} | Open set: {len(open_set)} | Current h: {current.h:.1f}")
            
            if (current.x, current.y) == goal:
                path = []
                while current:
                    path.append((current.x, current.y))
                    current = current.parent
                path = path[::-1]
                
                if debug:
                    print(f"\n✅ PATH FOUND!")
                    print(f"Total nodes explored: {nodes_explored}")
                    print(f"Path length: {len(path)} steps")
                    print(f"⚠️  Note: GBFS may not find shortest path")
                    print(f"First 10 steps: {path[:10]}")
                    print("="*60)
                
                return path
            
            closed_set.add((current.x, current.y))
            
            for nx, ny in PathfindingAlgorithms.get_neighbors(current, grid_width, grid_height):
                if (nx, ny) in closed_set or (nx, ny) in obstacles:
                    continue
                
                if (nx, ny) not in node_dict:
                    neighbor = Node(nx, ny)
                    neighbor.parent = current
                    neighbor.g = current.g + 1
                    neighbor.h = PathfindingAlgorithms.heuristic((nx, ny), goal)
                    neighbor.f = neighbor.h
                    node_dict[(nx, ny)] = neighbor
                    heapq.heappush(open_set, neighbor)
        
        if debug:
            print(f"\n❌ NO PATH FOUND!")
            print(f"Total nodes explored: {nodes_explored}")
            print("="*60)
        
        return None
    
    @staticmethod
    def dijkstra(start: Tuple[int, int], goal: Tuple[int, int], 
                 obstacles: set, grid_width: int, grid_height: int, debug: bool = True) -> Optional[List[Tuple[int, int]]]:
        """Dijkstra's Algorithm: Uses only g(n) cost from start"""
        if debug:
            print("\n" + "="*60)
            print("📐 DIJKSTRA'S ALGORITHM DEBUG")
            print("="*60)
            print(f"Start: {start} → Goal: {goal}")
            print(f"Obstacles: {len(obstacles)} towers placed")
            print("Strategy: Exploring uniformly from start - guaranteed shortest path")
        
        start_node = Node(start[0], start[1], g=0)
        start_node.f = 0
        
        open_set = []
        heapq.heappush(open_set, start_node)
        closed_set = set()
        node_dict = {(start[0], start[1]): start_node}
        
        nodes_explored = 0
        
        while open_set:
            current = heapq.heappop(open_set)
            nodes_explored += 1
            
            if debug and nodes_explored % 50 == 0:
                print(f"Nodes explored: {nodes_explored} | Open set: {len(open_set)} | Current cost: {current.g}")
            
            if (current.x, current.y) == goal:
                path = []
                while current:
                    path.append((current.x, current.y))
                    current = current.parent
                path = path[::-1]
                
                if debug:
                    print(f"\n✅ PATH FOUND!")
                    print(f"Total nodes explored: {nodes_explored}")
                    print(f"Path length: {len(path)} steps")
                    print(f"Path cost (guaranteed shortest): {len(path) - 1}")
                    print(f"First 10 steps: {path[:10]}")
                    print("="*60)
                
                return path
            
            closed_set.add((current.x, current.y))
            
            for nx, ny in PathfindingAlgorithms.get_neighbors(current, grid_width, grid_height):
                if (nx, ny) in closed_set or (nx, ny) in obstacles:
                    continue
                
                tentative_g = current.g + 1
                
                if (nx, ny) not in node_dict:
                    neighbor = Node(nx, ny)
                    node_dict[(nx, ny)] = neighbor
                else:
                    neighbor = node_dict[(nx, ny)]
                
                if tentative_g < neighbor.g:
                    neighbor.parent = current
                    neighbor.g = tentative_g
                    neighbor.f = neighbor.g
                    
                    if neighbor not in open_set:
                        heapq.heappush(open_set, neighbor)
        
        if debug:
            print(f"\n❌ NO PATH FOUND!")
            print(f"Total nodes explored: {nodes_explored}")
            print("="*60)
        
        return None
    
    @staticmethod
    def find_path(start: Tuple[int, int], goal: Tuple[int, int], 
                  obstacles: set, grid_width: int, grid_height: int, 
                  algorithm: str = ALGORITHM_ASTAR, debug: bool = True) -> Optional[List[Tuple[int, int]]]:
        """Unified pathfinding method that selects algorithm"""
        if algorithm == ALGORITHM_ASTAR:
            return PathfindingAlgorithms.astar(start, goal, obstacles, grid_width, grid_height, debug)
        elif algorithm == ALGORITHM_GBFS:
            return PathfindingAlgorithms.greedy_best_first(start, goal, obstacles, grid_width, grid_height, debug)
        elif algorithm == ALGORITHM_DIJKSTRA:
            return PathfindingAlgorithms.dijkstra(start, goal, obstacles, grid_width, grid_height, debug)
        else:
            return PathfindingAlgorithms.astar(start, goal, obstacles, grid_width, grid_height, debug)


class Bullet:
    def __init__(self, x: float, y: float, target: 'Enemy'):
        self.x = x
        self.y = y
        self.target = target
        self.speed = 8
        self.damage = 25
        self.active = True
    
    def update(self):
        if not self.target.active:
            self.active = False
            return
        
        dx = self.target.x - self.x
        dy = self.target.y - self.y
        dist = math.sqrt(dx**2 + dy**2)
        
        if dist < self.speed:
            self.target.take_damage(self.damage)
            self.active = False
        else:
            self.x += (dx / dist) * self.speed
            self.y += (dy / dist) * self.speed
    
    def draw(self, screen):
        pygame.draw.circle(screen, YELLOW, (int(self.x), int(self.y)), 4)


class Tower:
    def __init__(self, grid_x: int, grid_y: int):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.x = grid_x * GRID_SIZE + GRID_SIZE // 2
        self.y = grid_y * GRID_SIZE + GRID_SIZE // 2 + GAME_AREA_Y_OFFSET
        self.range = 150
        self.cooldown = 0
        self.max_cooldown = 30
    
    def update(self, enemies: List['Enemy'], bullets: List[Bullet]):
        if self.cooldown > 0:
            self.cooldown -= 1
            return
        
        for enemy in enemies:
            if enemy.active:
                dist = math.sqrt((enemy.x - self.x)**2 + (enemy.y - self.y)**2)
                if dist <= self.range:
                    bullets.append(Bullet(self.x, self.y, enemy))
                    self.cooldown = self.max_cooldown
                    break
    
    def draw(self, screen):
        pygame.draw.circle(screen, BLUE, (int(self.x), int(self.y)), 12)
        pygame.draw.circle(screen, DARK_GREEN, (int(self.x), int(self.y)), self.range, 1)


class Enemy:
    def __init__(self, path: List[Tuple[int, int]]):
        self.path = path
        self.path_index = 0
        self.x = path[0][0] * GRID_SIZE + GRID_SIZE // 2
        self.y = path[0][1] * GRID_SIZE + GRID_SIZE // 2 + GAME_AREA_Y_OFFSET
        self.speed = 1.5
        self.health = 1000
        self.max_health = 1000
        self.active = True
        self.reached_goal = False
        self.current_grid_pos = path[0] if path else None
        
        self.animation_frames = []
        self.current_frame = 0
        self.animation_speed = 10
        self.animation_counter = 0
        self.sprite_size = (32, 32)
        self.facing_right = True 
        
        self.load_animation_frames()
    
    def load_animation_frames(self):
        """Load sprite animation frames"""
        try:
            for i in range(1, 7):
                try:
                    frame = pygame.image.load(f"assets/enemy_walk_{i}.png")
                    frame = pygame.transform.scale(frame, self.sprite_size)
                    self.animation_frames.append(frame)
                except:
                    if i <= 4:
                        try:
                            frame = pygame.image.load(f"assets/enemy_walk_{i}.png")
                            frame = pygame.transform.scale(frame, self.sprite_size)
                            self.animation_frames.append(frame)
                        except:
                            pass
        except:
            pass
        
        if not self.animation_frames:
            print("⚠️  Enemy sprites not found, using fallback")
            for i in range(4):
                surface = pygame.Surface(self.sprite_size, pygame.SRCALPHA)
                radius = 10 + (i % 2) * 2
                pygame.draw.circle(surface, RED, (16, 16), radius)
                self.animation_frames.append(surface)
    
    def take_damage(self, damage: int):
        self.health -= damage
        if self.health <= 0:
            self.active = False
    
    def update(self):
        if not self.active or self.path_index >= len(self.path):
            return
        
        target_x = self.path[self.path_index][0] * GRID_SIZE + GRID_SIZE // 2
        target_y = self.path[self.path_index][1] * GRID_SIZE + GRID_SIZE // 2 + GAME_AREA_Y_OFFSET
        
        dx = target_x - self.x
        dy = target_y - self.y
        dist = math.sqrt(dx**2 + dy**2)
        
        # Update facing direction
        if dx > 0:
            self.facing_right = True
        elif dx < 0:
            self.facing_right = False
        
        if dist < self.speed:
            self.path_index += 1
            if self.path_index < len(self.path):
                self.current_grid_pos = self.path[self.path_index]
            if self.path_index >= len(self.path):
                self.reached_goal = True
                self.active = False
        else:
            self.x += (dx / dist) * self.speed
            self.y += (dy / dist) * self.speed
        
        # Update animation
        self.animation_counter += 1
        if self.animation_counter >= self.animation_speed:
            self.animation_counter = 0
            self.current_frame = (self.current_frame + 1) % len(self.animation_frames)
    
    def update_path(self, new_path: List[Tuple[int, int]]):
        if not new_path or not self.current_grid_pos:
            return
        
        min_dist = float('inf')
        best_index = 0
        
        for i, grid_pos in enumerate(new_path):
            dist = abs(grid_pos[0] - self.current_grid_pos[0]) + abs(grid_pos[1] - self.current_grid_pos[1])
            if dist < min_dist:
                min_dist = dist
                best_index = i
        
        self.path = new_path
        self.path_index = best_index
    
    def draw(self, screen):
        if self.animation_frames:
            current_sprite = self.animation_frames[self.current_frame]
            
            # Flip sprite based on direction
            if not self.facing_right:
                current_sprite = pygame.transform.flip(current_sprite, True, False)
            
            sprite_x = self.x - self.sprite_size[0] // 2
            sprite_y = self.y - self.sprite_size[1] // 2
            
            screen.blit(current_sprite, (sprite_x, sprite_y))
        else:
            pygame.draw.circle(screen, RED, (int(self.x), int(self.y)), 10)
        
        # Health bar
        bar_width = 30
        bar_height = 5
        bar_x = self.x - bar_width // 2
        bar_y = self.y - self.sprite_size[1] // 2 - 10
        
        pygame.draw.rect(screen, BLACK, (bar_x, bar_y, bar_width, bar_height))
        health_width = (self.health / self.max_health) * bar_width
        pygame.draw.rect(screen, GREEN, (bar_x, bar_y, health_width, bar_height))


class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.FULLSCREEN)
        pygame.display.set_caption("Tower Defense")
        self.clock = pygame.time.Clock()
        self.running = True
        
        self.state = STATE_MENU
        self.paused = False
        
        self.master_volume = 0.25
        pygame.mixer.music.set_volume(self.master_volume)
        
        # load fonts from assets
        self.font = self.load_font("assets/font.ttf", 40)
        self.small_font = self.load_font("assets/font.ttf", 28)
        self.title_font = self.load_font("assets/font.ttf", 64)
        
        # load background images
        self.menu_background = self.load_background("assets/menu-background.png")
        self.options_background = self.load_background("assets/option-background.png")
        self.game_background = self.load_background("assets/game-background.png")

        
        # menu buttons
        button_x = WINDOW_WIDTH // 2 - 150
        button_y_start = WINDOW_HEIGHT // 2 - 150
        button_spacing = 120
        
        self.menu_buttons = [
            Button(button_x, button_y_start, 
                   "assets/start-button.png", "assets/start-button-hover.png", 
                   self.start_game),
            Button(button_x, button_y_start + button_spacing * 1.35, 
                   "assets/option-button.png", "assets/option-button-hover.png", 
                   self.show_options),
            Button(button_x, button_y_start + button_spacing * 2.7, 
                   "assets/quit-button.png", "assets/quit-button-hover.png", 
                   self.quit_game)
        ]
        
        # options UI
        self.volume_slider_rect = pygame.Rect(WINDOW_WIDTH // 2 - 200, WINDOW_HEIGHT // 2, 400, 20)
        self.volume_handle_rect = pygame.Rect(0, 0, 20, 40)
        self.dragging_volume = False
        
        back_button_x = WINDOW_WIDTH // 2 - 150
        back_button_y = WINDOW_HEIGHT - 350
        self.back_button = Button(back_button_x, back_button_y,
                                   "assets/quit-button.png", "assets/quit-button-hover.png",
                                   self.show_menu)
        
        # game variables
        self.start_pos = (0, GRID_HEIGHT // 2)
        self.goal_pos = (GRID_WIDTH - 1, GRID_HEIGHT // 2)
        
        self.towers = []
        self.enemies = []
        self.bullets = []
        
        self.money = 500
        self.tower_cost = 50
        self.lives = 20
        self.score = 0
        
        self.wave = 0
        self.enemies_per_wave = 5
        self.spawn_timer = 0
        self.spawn_delay = 60
        self.enemies_spawned = 0
        self.wave_active = False
        
        self.selected_algorithm = ALGORITHM_ASTAR
        
        # UI buttons - Bottom bar
        button_width = 150
        button_height = 60
        button_spacing = 20
        right_offset = 50
        
        # algorithm selection buttons (bottom right)
        self.astar_button = UIButton(
            WINDOW_WIDTH - right_offset - (button_width * 4 + button_spacing * 21),
            WINDOW_HEIGHT - BOTTOM_BAR_HEIGHT + 20,
            button_width + 100, button_height, "A*", 
            lambda: self.select_algorithm(ALGORITHM_ASTAR), self.small_font
        )
        self.gbfs_button = UIButton(
            WINDOW_WIDTH - right_offset - (button_width * 3 + button_spacing * 14),
            WINDOW_HEIGHT - BOTTOM_BAR_HEIGHT + 20,
            button_width + 100, button_height, "G-BFS", 
            lambda: self.select_algorithm(ALGORITHM_GBFS), self.small_font
        )
        self.dijkstra_button = UIButton(
            WINDOW_WIDTH - right_offset - (button_width * 2 + button_spacing * 7),
            WINDOW_HEIGHT - BOTTOM_BAR_HEIGHT + 20,
            button_width + 100, button_height, "Dijkstra", 
            lambda: self.select_algorithm(ALGORITHM_DIJKSTRA), self.small_font
        )
        
        # start wave button (bottom right)
        self.start_wave_button = UIButton(
            WINDOW_WIDTH - right_offset - button_width,
            WINDOW_HEIGHT - BOTTOM_BAR_HEIGHT + 20,
            button_width + 25, button_height, "Start", self.start_wave, self.small_font
        )
        
        # menu button (top right)
        self.menu_button = UIButton(
            WINDOW_WIDTH - 150,
            20,
            140 , 60, "Menu", self.show_menu, self.small_font
        )
        
        self.algorithm_buttons = [self.astar_button, self.gbfs_button, self.dijkstra_button]
        self.astar_button.selected = True
    
    def select_algorithm(self, algorithm: str):
        """Select which pathfinding algorithm to use"""
        print(f"\n🔧 Switching algorithm to: {algorithm}")
        self.selected_algorithm = algorithm
        
        # update button selection states
        self.astar_button.selected = (algorithm == ALGORITHM_ASTAR)
        self.gbfs_button.selected = (algorithm == ALGORITHM_GBFS)
        self.dijkstra_button.selected = (algorithm == ALGORITHM_DIJKSTRA)
        
        # recalculate all enemy paths with new algorithm
        obstacles = self.get_obstacles()
        active_enemies = [e for e in self.enemies if e.active and e.current_grid_pos]
        
        if active_enemies:
            print(f"🔄 Recalculating paths for {len(active_enemies)} active enemies...")
            
        for enemy in active_enemies:
            new_path = PathfindingAlgorithms.find_path(
                enemy.current_grid_pos, self.goal_pos, 
                obstacles, GRID_WIDTH, GRID_HEIGHT, 
                self.selected_algorithm, debug=False
            )
            if new_path:
                enemy.update_path(new_path)
        
        if active_enemies:
            print(f"✅ All enemy paths updated with {algorithm}")
    
    def load_font(self, path: str, size: int):
        """Load custom font from assets, fallback to default if not found"""
        try:
            return pygame.font.Font(path, size)
        except:
            print(f"Font not found at {path}, using default font")
            return pygame.font.Font(None, size)
    
    def load_background(self, path: str):
        """Load and scale background image to fit window"""
        try:
            bg = pygame.image.load(path)
            bg = pygame.transform.scale(bg, (WINDOW_WIDTH, WINDOW_HEIGHT))
            return bg
        except:
            # Fallback: create gradient background
            bg = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
            for y in range(WINDOW_HEIGHT):
                color_value = int(20 + (y / WINDOW_HEIGHT) * 40)
                pygame.draw.line(bg, (color_value, color_value, color_value), (0, y), (WINDOW_WIDTH, y))
            return bg
    
    # menu actions
    def start_game(self):
        self.state = STATE_GAME
        self.reset_game()
    
    def show_options(self):
        self.state = STATE_OPTIONS
    
    def show_menu(self):
        self.state = STATE_MENU
    
    def quit_game(self):
        self.running = False
    
    def reset_game(self):
        self.towers = []
        self.enemies = []
        self.bullets = []
        self.money = 500
        self.lives = 20
        self.score = 0
        self.wave = 0
        self.enemies_spawned = 0
        self.wave_active = False
        self.paused = False
        self.selected_algorithm = ALGORITHM_ASTAR
        self.astar_button.selected = True
        self.gbfs_button.selected = False
        self.dijkstra_button.selected = False
    
    # Game actions
    def start_wave(self):
        if not self.wave_active:
            self.wave_active = True
            self.wave += 1
            self.enemies_per_wave = 5 + self.wave * 2
            self.enemies_spawned = 0
            self.spawn_timer = 0
    
    def toggle_pause(self):
        self.paused = not self.paused
    
    def get_obstacles(self) -> set:
        obstacles = set()
        for tower in self.towers:
            obstacles.add((tower.grid_x, tower.grid_y))
        return obstacles
    
    def spawn_enemy(self):
        obstacles = self.get_obstacles()
        print(f"\n🎮 Spawning enemy #{self.enemies_spawned + 1} for Wave {self.wave}")
        path = PathfindingAlgorithms.find_path(
            self.start_pos, self.goal_pos, obstacles, 
            GRID_WIDTH, GRID_HEIGHT, self.selected_algorithm, debug=True
        )
        
        if path:
            self.enemies.append(Enemy(path))
            self.enemies_spawned += 1
            print(f"✅ Enemy spawned successfully!")
        else:
            print(f"❌ Failed to spawn enemy - no path available!")
    
    def place_tower(self, grid_x: int, grid_y: int):
        if self.money < self.tower_cost:
            print(f"❌ Cannot place tower: Insufficient funds (${self.money} < ${self.tower_cost})")
            return False
        
        if (grid_x, grid_y) == self.start_pos or (grid_x, grid_y) == self.goal_pos:
            print(f"❌ Cannot place tower: Blocking start/goal position")
            return False
        
        for tower in self.towers:
            if tower.grid_x == grid_x and tower.grid_y == grid_y:
                print(f"❌ Cannot place tower: Position already occupied")
                return False
        
        test_obstacles = self.get_obstacles()
        test_obstacles.add((grid_x, grid_y))
        
        print(f"\n🏗️  Testing tower placement at ({grid_x}, {grid_y})")
        path = PathfindingAlgorithms.find_path(
            self.start_pos, self.goal_pos, test_obstacles, 
            GRID_WIDTH, GRID_HEIGHT, self.selected_algorithm, debug=True
        )
        
        if not path:
            print(f"❌ Tower placement blocked - would block all paths!")
            return False
        
        self.towers.append(Tower(grid_x, grid_y))
        self.money -= self.tower_cost
        print(f"✅ Tower placed! Remaining money: ${self.money}")
        
        # Recalculate paths for active enemies
        obstacles = self.get_obstacles()
        enemies_updated = 0
        for enemy in self.enemies:
            if enemy.active and enemy.current_grid_pos:
                print(f"\n🔄 Recalculating path for enemy at {enemy.current_grid_pos}")
                new_path = PathfindingAlgorithms.find_path(
                    enemy.current_grid_pos, self.goal_pos, 
                    obstacles, GRID_WIDTH, GRID_HEIGHT, 
                    self.selected_algorithm, debug=False
                )
                if new_path:
                    enemy.update_path(new_path)
                    enemies_updated += 1
        
        if enemies_updated > 0:
            print(f"✅ Updated paths for {enemies_updated} active enemies")
        
        return True
    
    def update_game(self):
        if self.paused:
            return
        
        # spawn enemies during active wave
        if self.wave_active and self.enemies_spawned < self.enemies_per_wave:
            self.spawn_timer += 1
            if self.spawn_timer >= self.spawn_delay:
                self.spawn_enemy()
                self.spawn_timer = 0
        
        # check if wave is complete
        if self.wave_active and self.enemies_spawned >= self.enemies_per_wave:
            if all(not e.active for e in self.enemies):
                self.wave_active = False
        
        for enemy in self.enemies:
            if enemy.active:
                enemy.update()
                if enemy.reached_goal:
                    self.lives -= 1
        
        self.enemies = [e for e in self.enemies if e.active or e.reached_goal]
        
        for tower in self.towers:
            tower.update(self.enemies, self.bullets)
        
        for bullet in self.bullets:
            bullet.update()
        
        self.bullets = [b for b in self.bullets if b.active]
        
        defeated_enemies = [e for e in self.enemies if not e.active and not e.reached_goal]
        for _ in defeated_enemies:
            self.money += 25
            self.score += 10
        
        self.enemies = [e for e in self.enemies if e.active]
    
    def draw_top_bar(self):
        """Draw the top UI bar with Score, Health, Coin"""
        # Background
        # pygame.draw.rect(self.screen, UI_BACKGROUND, (0, 0, WINDOW_WIDTH, TOP_BAR_HEIGHT))
        # pygame.draw.line(self.screen, BLACK, (0, TOP_BAR_HEIGHT), (WINDOW_WIDTH, TOP_BAR_HEIGHT), 3)
        
        # Left side stats
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        health_text = self.font.render(f"Health: {self.lives}", True, WHITE)
        coin_text = self.font.render(f"Coin: ${self.money}", True, WHITE)
        
        self.screen.blit(score_text, (50, 30))
        self.screen.blit(health_text, (50 + 400, 30))
        self.screen.blit(coin_text, (50 + 950, 30))

        # Algorithm indicator
        # algo_text = self.small_font.render(f"Algorithm: {self.selected_algorithm}", True, PURPLE)
        # self.screen.blit(algo_text, (50 + 1300, 30))
        
        # Menu button (top right)
        mouse_pos = pygame.mouse.get_pos()
        self.menu_button.update(mouse_pos)
        self.menu_button.draw(self.screen)
    
    def draw_bottom_bar(self):
        """Draw the bottom UI bar with Wave and algorithm buttons"""
        # Background
        # pygame.draw.rect(self.screen, UI_BACKGROUND, (0, WINDOW_HEIGHT - BOTTOM_BAR_HEIGHT, WINDOW_WIDTH, BOTTOM_BAR_HEIGHT))
        # pygame.draw.line(self.screen, BLACK, (0, WINDOW_HEIGHT - BOTTOM_BAR_HEIGHT), (WINDOW_WIDTH, WINDOW_HEIGHT - BOTTOM_BAR_HEIGHT), 3)
        
        # Wave counter (left side)
        wave_text = self.font.render(f"Wave: {self.wave}", True, BLACK)
        self.screen.blit(wave_text, (50, WINDOW_HEIGHT - BOTTOM_BAR_HEIGHT + 30))
        
        # Algorithm selection buttons and start button (right side)
        mouse_pos = pygame.mouse.get_pos()
        
        for button in self.algorithm_buttons:
            button.update(mouse_pos)
            button.draw(self.screen)
        
        if not self.wave_active and all(not e.active for e in self.enemies):
            self.start_wave_button.update(mouse_pos)
            self.start_wave_button.draw(self.screen)
    
    def draw_menu(self):
        # Draw background
        self.screen.blit(self.menu_background, (0, 0))
        
        mouse_pos = pygame.mouse.get_pos()
        for button in self.menu_buttons:
            button.update(mouse_pos)
            button.draw(self.screen)
    
    def draw_options(self):
        # Draw background
        self.screen.blit(self.options_background, (0, 0))
        
        # Volume label
        volume_label = self.small_font.render("Master Volume", True, WHITE)
        label_rect = volume_label.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 50))
        self.screen.blit(volume_label, label_rect)
        
        # Volume slider
        pygame.draw.rect(self.screen, GRAY, self.volume_slider_rect)
        
        # Volume handle
        handle_x = self.volume_slider_rect.x + int(self.master_volume * self.volume_slider_rect.width) - 10
        self.volume_handle_rect.center = (handle_x, self.volume_slider_rect.centery)
        pygame.draw.rect(self.screen, BLUE, self.volume_handle_rect, border_radius=5)
        
        # Volume percentage
        volume_text = self.small_font.render(f"{int(self.master_volume * 100)}%", True, WHITE)
        volume_rect = volume_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 50))
        self.screen.blit(volume_text, volume_rect)
        
        # Back button
        mouse_pos = pygame.mouse.get_pos()
        self.back_button.update(mouse_pos)
        self.back_button.draw(self.screen)
    
    def draw_game(self):
        self.screen.blit(self.game_background, (0, 0))
        
        # Draw top bar
        self.draw_top_bar()
        
        # Draw bottom bar
        self.draw_bottom_bar()
        
        # Draw game area
        # game_area_rect = pygame.Rect(0, TOP_BAR_HEIGHT, WINDOW_WIDTH, GAME_AREA_HEIGHT)
        # pygame.draw.rect(self.screen, WHITE, game_area_rect)
        
        # Draw hover preview in game area
        mouse_x, mouse_y = pygame.mouse.get_pos()
        
        # Check if mouse is in game area
        if TOP_BAR_HEIGHT <= mouse_y < WINDOW_HEIGHT - BOTTOM_BAR_HEIGHT:
            grid_x = mouse_x // GRID_SIZE
            grid_y = (mouse_y - GAME_AREA_Y_OFFSET) // GRID_SIZE
            
            if 0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT:
                hover_rect = pygame.Rect(grid_x * GRID_SIZE, 
                                         grid_y * GRID_SIZE + GAME_AREA_Y_OFFSET, 
                                         GRID_SIZE, GRID_SIZE)
                
                can_place = True
                
                if (grid_x, grid_y) == self.start_pos or (grid_x, grid_y) == self.goal_pos:
                    can_place = False
                
                for tower in self.towers:
                    if tower.grid_x == grid_x and tower.grid_y == grid_y:
                        can_place = False
                        break
                
                if self.money < self.tower_cost:
                    can_place = False
                
                hover_surface = pygame.Surface((GRID_SIZE, GRID_SIZE))
                hover_surface.set_alpha(120)
                if can_place:
                    hover_surface.fill(BLUE)
                else:
                    hover_surface.fill(RED)
                self.screen.blit(hover_surface, hover_rect)
                
                border_color = BLUE if can_place else RED
                pygame.draw.rect(self.screen, border_color, hover_rect, 2)
        
        # Draw start and goal
        start_rect = pygame.Rect(self.start_pos[0] * GRID_SIZE, 
                                 self.start_pos[1] * GRID_SIZE + GAME_AREA_Y_OFFSET, 
                                 GRID_SIZE, GRID_SIZE)
        pygame.draw.rect(self.screen, GREEN, start_rect)
        
        # Draw "enemy start" label
        # start_label = self.small_font.render("enemy start", True, BLACK)
        # self.screen.blit(start_label, (self.start_pos[0] * GRID_SIZE + 5, 
        #                                self.start_pos[1] * GRID_SIZE + GAME_AREA_Y_OFFSET - 30))
        
        goal_rect = pygame.Rect(self.goal_pos[0] * GRID_SIZE, 
                               self.goal_pos[1] * GRID_SIZE + GAME_AREA_Y_OFFSET, 
                               GRID_SIZE, GRID_SIZE)
        pygame.draw.rect(self.screen, PURPLE, goal_rect)
        
        # Draw "enemy finish" label
        # finish_label = self.small_font.render("enemy finish", True, BLACK)
        # self.screen.blit(finish_label, (self.goal_pos[0] * GRID_SIZE - 100, 
        #                                 self.goal_pos[1] * GRID_SIZE + GAME_AREA_Y_OFFSET - 30))
        
        # Draw "tower placement grid" label at center
        # grid_label = self.small_font.render("tower placement grid", True, GRAY)
        # label_rect = grid_label.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
        # self.screen.blit(grid_label, label_rect)
        
        # Draw game entities
        for tower in self.towers:
            tower.draw(self.screen)
        
        for bullet in self.bullets:
            bullet.draw(self.screen)
        
        for enemy in self.enemies:
            if enemy.active:
                enemy.draw(self.screen)
        
        # Game over
        if self.lives <= 0:
            overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
            overlay.set_alpha(200)
            overlay.fill(BLACK)
            self.screen.blit(overlay, (0, 0))
            
            game_over_text = self.title_font.render("GAME OVER!", True, RED)
            game_over_rect = game_over_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 50))
            self.screen.blit(game_over_text, game_over_rect)
            
            final_score_text = self.font.render(f"Final Score: {self.score}", True, WHITE)
            score_rect = final_score_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 50))
            self.screen.blit(final_score_text, score_rect)
            
            menu_text = self.small_font.render("Press ESC for Menu", True, WHITE)
            menu_rect = menu_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 120))
            self.screen.blit(menu_text, menu_rect)
    
    def handle_menu_input(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_pos = pygame.mouse.get_pos()
                for button in self.menu_buttons:
                    if button.is_clicked(mouse_pos):
                        button.action()
    
    def handle_options_input(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_pos = pygame.mouse.get_pos()
                if self.back_button.is_clicked(mouse_pos):
                    self.back_button.action()
                elif self.volume_handle_rect.collidepoint(mouse_pos):
                    self.dragging_volume = True
        
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.dragging_volume = False
        
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging_volume:
                mouse_x = pygame.mouse.get_pos()[0]
                slider_x = mouse_x - self.volume_slider_rect.x
                slider_x = max(0, min(slider_x, self.volume_slider_rect.width))
                self.master_volume = slider_x / self.volume_slider_rect.width
                pygame.mixer.music.set_volume(self.master_volume)
    
    def handle_game_input(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_pos = pygame.mouse.get_pos()
                mouse_x, mouse_y = mouse_pos
                
                # Check menu button
                if self.menu_button.is_clicked(mouse_pos):
                    self.menu_button.action()
                    return
                
                # Check algorithm selection buttons
                for button in self.algorithm_buttons:
                    if button.is_clicked(mouse_pos):
                        button.action()
                        return
                
                # Check start wave button
                if not self.wave_active and all(not e.active for e in self.enemies):
                    if self.start_wave_button.is_clicked(mouse_pos):
                        self.start_wave_button.action()
                        return
                
                # Place tower in game area
                if TOP_BAR_HEIGHT <= mouse_y < WINDOW_HEIGHT - BOTTOM_BAR_HEIGHT:
                    if not self.paused:
                        grid_x = mouse_x // GRID_SIZE
                        grid_y = (mouse_y - GAME_AREA_Y_OFFSET) // GRID_SIZE
                        self.place_tower(grid_x, grid_y)
    
    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.state == STATE_GAME and self.lives <= 0:
                            self.show_menu()
                        elif self.state == STATE_GAME:
                            self.show_menu()
                        else:
                            self.running = False
            
                if self.state == STATE_MENU:
                    self.handle_menu_input(event)
                elif self.state == STATE_OPTIONS:
                    self.handle_options_input(event)
                elif self.state == STATE_GAME:
                    self.handle_game_input(event)
            
            # Update and draw based on state
            if self.state == STATE_MENU:
                self.draw_menu()
            elif self.state == STATE_OPTIONS:
                self.draw_options()
            elif self.state == STATE_GAME:
                self.update_game()
                self.draw_game()
            
            pygame.display.flip()
            self.clock.tick(FPS)
        
        pygame.quit()


if __name__ == "__main__":
    game = Game()
    game.run()
