import pygame
import sys

# Initialize Pygame
pygame.init()

# Constants
SCREEN_SIZE = 300
TILE_SIZE = SCREEN_SIZE // 3
BACKGROUND_COLOR = (255, 255, 255)
FONT_COLOR = (0, 0, 0)
FONT_SIZE = 40

# Set up the display
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption("8 Puzzle Game")
font = pygame.font.Font(None, FONT_SIZE)

# Puzzle state
# Using 0 to represent the empty tile
puzzle_state = [[1, 2, 3],
                [4, 5, 6],
                [7, 8, 0]]


def find_empty_tile(state):
    for y, row in enumerate(state):
        for x, tile in enumerate(row):
            if tile == 0:
                return x, y
    return None  # Shouldn't happen


def move_tile(state, direction):
    x, y = find_empty_tile(state)
    if direction == "UP" and y < 2:
        state[y][x], state[y + 1][x] = state[y + 1][x], state[y][x]
    elif direction == "DOWN" and y > 0:
        state[y][x], state[y - 1][x] = state[y - 1][x], state[y][x]
    elif direction == "LEFT" and x < 2:
        state[y][x], state[y][x + 1] = state[y][x + 1], state[y][x]
    elif direction == "RIGHT" and x > 0:
        state[y][x], state[y][x - 1] = state[y][x - 1], state[y][x]


def draw_puzzle(state):
    for y in range(3):
        for x in range(3):
            tile = state[y][x]
            if tile != 0:
                # Draw tile
                rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                pygame.draw.rect(screen, FONT_COLOR, rect, 3)  # Tile border
                # Draw tile number
                text_surf = font.render(str(tile), True, FONT_COLOR)
                text_rect = text_surf.get_rect(center=rect.center)
                screen.blit(text_surf, text_rect)


def main():
    running = True
    clock = pygame.time.Clock()  # Clock object for controlling FPS
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    move_tile(puzzle_state, "UP")
                elif event.key == pygame.K_DOWN:
                    move_tile(puzzle_state, "DOWN")
                elif event.key == pygame.K_LEFT:
                    move_tile(puzzle_state, "LEFT")
                elif event.key == pygame.K_RIGHT:
                    move_tile(puzzle_state, "RIGHT")

        # Fill the background
        screen.fill(BACKGROUND_COLOR)

        # Draw the puzzle
        draw_puzzle(puzzle_state)

        pygame.display.flip()

        clock.tick(60)  # Limit the frame rate to 60 FPS

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
