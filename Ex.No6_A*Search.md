# Ex.No: 6  Implementation of Zombie survival game using A* search 
### DATE: 05/10/2024                                                                           
### REGISTER NUMBER : 212221240041
### AIM: 
To write a python program to simulate the Zomibie Survival game using A* Search 
### Algorithm:
1. Start the program
2. Import the necessary modules
3. Initiate the pygame engine and window
4. Collect the Zombie image and resize it within a display window 
5. Create a Euclidean distance heuristic function to find the distance from current location to Target position
6.  Move the Zombie towards the target by A* search 
7.  In main, create the obstacles and move the player by Key movements up, down,left and right 
10.  Update the display every time 
11.  Stop the program
 ### Program:

```
import pygame
import heapq
import random
import math
import time

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
PLAYER_RADIUS = 15
ZOMBIE_IMAGE_SIZE = 40  # Size of the zombie image (make sure the image is square)

# Set up the display
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Zombie Survival with A*")

# Load images
zombie_img = pygame.image.load('z1.png')
zombie_img = pygame.transform.scale(zombie_img, (ZOMBIE_IMAGE_SIZE, ZOMBIE_IMAGE_SIZE))

# Euclidean distance heuristic function
def heuristic(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

# A* Algorithm Implementation
def astar(start, goal, obstacles):
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_list:
        current = heapq.heappop(open_list)[1]

        if heuristic(current, goal) < ZOMBIE_IMAGE_SIZE / 2:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < WIDTH and 0 <= neighbor[1] < HEIGHT:
                if neighbor in obstacles:
                    continue
                tentative_g_score = g_score[current] + heuristic(current, neighbor)
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))

    return []  # Return an empty path if no path is found

# Draw Function
def draw(win, player_pos, zombies, obstacles, elapsed_time):
    win.fill(WHITE)
    pygame.draw.circle(win, GREEN, (int(player_pos[0]), int(player_pos[1])), PLAYER_RADIUS)
    for zombie in zombies:
        win.blit(zombie_img, (int(zombie[0] - ZOMBIE_IMAGE_SIZE / 2), int(zombie[1] - ZOMBIE_IMAGE_SIZE / 2)))
    for obs in obstacles:
        pygame.draw.rect(win, BLACK, pygame.Rect(obs[0], obs[1], 10, 10))
    
    # Display elapsed time
    font = pygame.font.SysFont(None, 36)
    time_text = font.render(f"Time: {elapsed_time:.2f} seconds", True, BLACK)
    win.blit(time_text, (10, 10))
    
    pygame.display.update()

# Main Game Loop
def main():
    clock = pygame.time.Clock()
    player_pos = [WIDTH // 2, HEIGHT // 2]
    zombies = [[random.randint(ZOMBIE_IMAGE_SIZE // 2, WIDTH - ZOMBIE_IMAGE_SIZE // 2), 
                random.randint(ZOMBIE_IMAGE_SIZE // 2, HEIGHT - ZOMBIE_IMAGE_SIZE // 2)] 
               for _ in range(5)]
    obstacles = [(random.randint(0, WIDTH-10), random.randint(0, HEIGHT-10)) for _ in range(20)]

    start_time = time.time()  # Record start time

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            player_pos[0] -= 5
        if keys[pygame.K_RIGHT]:
            player_pos[0] += 5
        if keys[pygame.K_UP]:
            player_pos[1] -= 5
        if keys[pygame.K_DOWN]:
            player_pos[1] += 5

        # Update zombies' positions
        new_zombies = []
        for zombie in zombies:
            path = astar(tuple(zombie), tuple(player_pos), set(obstacles))
            if path:
                next_step = path[1] if len(path) > 1 else path[0]
                direction = [next_step[0] - zombie[0], next_step[1] - zombie[1]]
                magnitude = math.hypot(*direction)
                if magnitude > 0:
                    direction = [direction[0] / magnitude, direction[1] / magnitude]
                    new_zombies.append([zombie[0] + direction[0] * 2, zombie[1] + direction[1] * 2])
                else:
                    new_zombies.append(zombie)
            else:
                new_zombies.append(zombie)
        zombies = new_zombies

        elapsed_time = time.time() - start_time  # Calculate elapsed time

        draw(win, player_pos, zombies, obstacles, elapsed_time)
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()

```
### Output:
![a](https://github.com/user-attachments/assets/87137f09-eb0e-413b-ad10-e9d67d34333a)



### Result:
Thus the simple Zombie survival game was implemented using python.
