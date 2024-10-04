
# Ex.No: 11  Mini Project 
DATE: 10/412024                                                                           
REGISTER NUMBER : 212221240041
### AIM: 
The aim of this project is to create an AI-powered car racing game where the player and an AI-controlled car avoid obstacles (bombs) 
on a scrolling road. The AI uses Q-learning to learn and improve its performance over time, trying to avoid collisions with the obstacles.
### Algorithm:
1.Initialize Pygame:

Start the game engine and set up the screen, images, and game settings.
2.Define Game Environment:

Set the screen dimensions, car size, speed, and obstacle characteristics (size, speed, frequency).
Load the player’s car, AI car, and obstacle images.
Set up a scrolling background effect.
3.Q-learning Setup:

Define an action space for the AI car: Move Left, Stay, Move Right.
Set up a simplified state space based on the AI car’s position and the closest obstacle’s position.
Initialize the Q-table, which stores the expected rewards for different actions in each state.
Set learning rate, discount factor, exploration rate (for exploration-exploitation trade-off), and exploration decay rate.

### Program:
```
import pygame
import random
import math
import numpy as np

# Initialize Pygame
pygame.init()

# Screen settings
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))

pygame.display.set_caption("AI Car Racing Game")

# Game settings
player_car_width, player_car_height = 100, 100  # Set player car size
ai_car_width, ai_car_height = 100, 100           # Set AI car size
car_speed = 5
dodge_speed = 15
center_x = WIDTH // 2 - player_car_width // 2

# Q-learning settings
action_space = [-1, 0, 1]  # -1: Move Left, 0: Stay, 1: Move Right
num_actions = len(action_space)
state_space_size = (WIDTH // car_speed, WIDTH // car_speed)  # Simplified state space (AI_x, Closest_obstacle_x)
q_table = np.zeros(state_space_size + (num_actions,))  # Initialize Q-table with zeros

learning_rate = 0.1
discount_factor = 0.95
exploration_rate = 1.0
exploration_decay = 0.995
min_exploration_rate = 0.01

# Obstacle settings
obstacle_radius = 30  # Set the radius for bombs
obstacle_min_speed = 5
obstacle_max_speed = 10
obstacle_frequency = 1500
last_obstacle_time = pygame.time.get_ticks()

# Load images
player_car_img = pygame.image.load('carme.png')
player_car_img = pygame.transform.scale(player_car_img, (player_car_width, player_car_height))

ai_car_img = pygame.image.load('carAI.png')
ai_car_img = pygame.transform.scale(ai_car_img, (ai_car_width, ai_car_height))

bomb_img = pygame.image.load('bomb.png')
bomb_img = pygame.transform.scale(bomb_img, (obstacle_radius * 2, obstacle_radius * 2))

# Background
background_img = pygame.image.load('back.jpg')
background_img = pygame.transform.scale(background_img, (WIDTH, HEIGHT))
background_y1 = 0
background_y2 = -HEIGHT

# Scores
player_score = 0
ai_score = 0
font = pygame.font.Font(None, 36)

# Reset game variables
def reset_game():
    global player_x, player_y, ai_x, ai_y, player_score, ai_score, obstacles, exploration_rate, q_table
    player_x = WIDTH // 2 - player_car_width // 2
    player_y = HEIGHT - player_car_height - 20
    ai_x = WIDTH // 2 + ai_car_width // 2
    ai_y = HEIGHT - ai_car_height - 20
    player_score = 0
    ai_score = 0
    exploration_rate = 1.0
    q_table = np.zeros(state_space_size + (num_actions,))
    obstacles = []

# AI movement logic with Q-learning
def ai_move(ai_x, ai_y, obstacles):
    global exploration_rate, ai_score

    # Get the closest obstacle
    closest_obstacle = None
    min_distance = HEIGHT
    for obstacle in obstacles:
        obstacle_x, obstacle_y, _, _, _ = obstacle
        distance = ai_y - obstacle_y
        if distance > 0 and distance < min_distance:
            min_distance = distance
            closest_obstacle = obstacle

    # Define AI state
    if closest_obstacle:
        obstacle_x = closest_obstacle[0]
    else:
        obstacle_x = center_x

    ai_state = (ai_x // car_speed, obstacle_x // car_speed)

    # Q-learning action selection (epsilon-greedy)
    if np.random.uniform(0, 1) < exploration_rate:
        action = np.random.choice(num_actions)  # Random action (exploration)
    else:
        action = np.argmax(q_table[ai_state])  # Best action (exploitation)

    # Take action and update AI car position
    ai_x += action_space[action] * dodge_speed

    # Ensure AI doesn't go off screen
    ai_x = max(0, min(ai_x, WIDTH - ai_car_width))

    # Get the reward
    reward = get_reward(ai_x, ai_y, closest_obstacle)

    # Update Q-table
    new_ai_state = (ai_x // car_speed, obstacle_x // car_speed)
    q_table[ai_state][action] = q_table[ai_state][action] + learning_rate * (reward + discount_factor * np.max(q_table[new_ai_state]) - q_table[ai_state][action])

    # Exploration decay
    exploration_rate = max(min_exploration_rate, exploration_rate * exploration_decay)

    return ai_x

# Get reward based on AI position and obstacles
def get_reward(ai_x, ai_y, closest_obstacle):
    if closest_obstacle:
        obstacle_x, obstacle_y, _, _, _ = closest_obstacle
        distance = math.sqrt((ai_x - obstacle_x) ** 2 + (ai_y - obstacle_y) ** 2)
        if distance < obstacle_radius + ai_car_width // 2:  # AI hit obstacle
            return -100  # Negative reward for collision
        elif distance < 100:  # AI dodged obstacle
            return 10  # Positive reward for dodging
    return -1  # Small negative reward for each frame to encourage speed

# Circle collision
def circle_collision(x1, y1, r1, x2, y2, r2):
    distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return distance < (r1 + r2)

# Display the retry screen
def show_retry_screen():
    screen.fill((255, 255, 255))  # Fill the screen with white
    retry_text = font.render("Game Over! Press 'R' to Retry or 'Q' to Quit", True, (0, 0, 0))
    screen.blit(retry_text, (WIDTH // 2 - retry_text.get_width() // 2, HEIGHT // 2))
    pygame.display.update()

    # Wait for the player's input
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    reset_game()  # Reset the game
                    waiting = False  # Exit the waiting loop
                elif event.key == pygame.K_q:
                    pygame.quit()
                    quit()

# Game loop settings
clock = pygame.time.Clock()
FPS = 60
running = True

# Reset the game before starting
reset_game()

# Main loop
while running:
    # Draw scrolling background
    screen.blit(background_img, (0, background_y1))
    screen.blit(background_img, (0, background_y2))

    background_y1 += 5
    background_y2 += 5

    if background_y1 >= HEIGHT:
        background_y1 = -HEIGHT
    if background_y2 >= HEIGHT:
        background_y2 = -HEIGHT

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Player car movement
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and player_x > 0:
        player_x -= car_speed
    if keys[pygame.K_RIGHT] and player_x < WIDTH - player_car_width:
        player_x += car_speed

    # Generate obstacles
    current_time = pygame.time.get_ticks()
    if current_time - last_obstacle_time > obstacle_frequency:
        obstacle_x = random.randint(0 + obstacle_radius, WIDTH - obstacle_radius * 2)
        obstacle_speed = random.randint(obstacle_min_speed, obstacle_max_speed)
        obstacles.append([obstacle_x, -obstacle_radius * 2, obstacle_radius, obstacle_radius * 2, obstacle_speed])
        last_obstacle_time = current_time

    # Move obstacles
    for obstacle in obstacles:
        obstacle[1] += obstacle[4]
        screen.blit(bomb_img, (obstacle[0], obstacle[1]))

    # Remove off-screen obstacles
    obstacles = [ob for ob in obstacles if ob[1] < HEIGHT]

    # AI movement with Q-learning
    ai_x = ai_move(ai_x, ai_y, obstacles)

    # Draw player and AI cars
    screen.blit(player_car_img, (player_x, player_y))
    screen.blit(ai_car_img, (ai_x, ai_y))

    # Collision detection (player)
    player_center = (player_x + player_car_width // 2, player_y + player_car_height // 2)
    player_radius = player_car_width // 2 * 0.5
    for obstacle in obstacles:
        obstacle_center = (obstacle[0] + obstacle_radius, obstacle[1] + obstacle_radius)
        if circle_collision(player_center[0], player_center[1], player_radius, obstacle_center[0], obstacle_center[1], obstacle_radius):
            print("Game Over!")
            show_retry_screen()  # Show retry screen when game over

    # Collision detection (AI)
    ai_center = (ai_x + ai_car_width // 2, ai_y + ai_car_height // 2)
    ai_radius = ai_car_width // 2 * 0.5
    for obstacle in obstacles:
        obstacle_center = (obstacle[0] + obstacle_radius, obstacle[1] + obstacle_radius)
        if circle_collision(ai_center[0], ai_center[1], ai_radius, obstacle_center[0], obstacle_center[1], obstacle_radius):
            print("AI crashed!")
            show_retry_screen()  # Show retry screen when AI crashes

    # Update the display
    pygame.display.update()
    clock.tick(FPS)

pygame.quit()


```
### Output:



![a](https://github.com/user-attachments/assets/e81a7191-67b6-483a-884c-bcd064e2a1e3)
![b](https://github.com/user-attachments/assets/3f151ed9-4f0d-4e27-80aa-a1849b79610e)

## Result:
The AI car racing game uses Q-learning to train the AI to dodge obstacles while the player manually controls their own car to avoid collisions.


