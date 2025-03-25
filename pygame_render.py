import pygame
import json

# Constants
GRID_SIZE = 40  # Cell size
MARGIN = 2  # Spacing
SCREEN_SIZE = (6, 6)  # Match environment size
FONT_SIZE = 24

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 50, 50)
GREEN = (50, 200, 50)
GRAY = (100, 100, 100)


def load_environment(filename="environment.json"):
    """Loads environment data and converts lists to tuples."""
    with open(filename, "r") as file:
        data = json.load(file)

    data["obstacles"] = [tuple(pos) for pos in data["obstacles"]]
    data["red_cells"] = [tuple(pos) for pos in data["red_cells"]]
    data["green_cells"] = [tuple(pos) for pos in data["green_cells"]]

    return data


def load_gameplay_log(episode_number, filename="gameplay_log.json"):
    """Loads gameplay trajectory for a specific episode and converts lists to tuples."""
    with open(filename, "r") as file:
        data = json.load(file)

    episode_key = str(episode_number)
    if episode_key not in data:
        raise ValueError(f"Episode {episode_number} not found in the log!")

    turns = data[episode_key]["turns"]

    for turn in turns:
        turn["p1t"] = tuple(turn["p1t"])
        turn["p2t"] = tuple(turn["p2t"])

    return turns


def draw_grid(screen, width, height, env_data, p1_pos, p2_pos, drone_img, robot_img, turn_num, font):
    """Draws the environment grid, players, and turn number."""
    screen.fill(WHITE)  # Background color

    for x in range(width):
        for y in range(height):
            rect = pygame.Rect(x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE - MARGIN, GRID_SIZE - MARGIN)

            if (x, y) in env_data["obstacles"]:
                pygame.draw.rect(screen, GRAY, rect)  # Obstacles
            elif (x, y) in env_data["red_cells"]:
                pygame.draw.rect(screen, RED, rect)  # Red Cells
            elif (x, y) in env_data["green_cells"]:
                pygame.draw.rect(screen, GREEN, rect)  # Green Cells
            else:
                pygame.draw.rect(screen, BLACK, rect, 1)  # Grid outline

    # Draw players using images
    screen.blit(drone_img, (p1_pos[0] * GRID_SIZE, p1_pos[1] * GRID_SIZE))
    screen.blit(robot_img, (p2_pos[0] * GRID_SIZE, p2_pos[1] * GRID_SIZE))

    # Display turn number
    turn_text = font.render(f"Turn: {turn_num}", True, BLACK)
    screen.blit(turn_text, (10, 10))


def run_episode(episode_number):
    """Runs a specific episode visualization in Pygame."""
    pygame.init()

    width, height = SCREEN_SIZE
    screen = pygame.display.set_mode((width * GRID_SIZE, height * GRID_SIZE))
    pygame.display.set_caption(f"Gameplay Trajectory - Episode {episode_number}")

    # Load assets
    env_data = load_environment("environment.json")
    turns = load_gameplay_log(episode_number, "pygame_render_debug.json")

    # Load images
    drone_img = pygame.image.load("drone.png")
    drone_img = pygame.transform.scale(drone_img, (GRID_SIZE - MARGIN, GRID_SIZE - MARGIN))

    robot_img = pygame.image.load("robot.png")
    robot_img = pygame.transform.scale(robot_img, (GRID_SIZE - MARGIN, GRID_SIZE - MARGIN))

    # Load font
    font = pygame.font.Font(None, FONT_SIZE)

    running = True
    step = 0  # Current turn index

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                if step < len(turns) - 1:
                    step += 1  # Move to next turn

        # Get current player positions
        p1_pos = turns[step]["p1t"]
        p2_pos = turns[step]["p2t"]

        # Draw everything
        draw_grid(screen, width, height, env_data, p1_pos, p2_pos, drone_img, robot_img, step + 1, font)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    episode_number = int(input("Enter episode number: "))  # Get episode from user
    run_episode(episode_number)
