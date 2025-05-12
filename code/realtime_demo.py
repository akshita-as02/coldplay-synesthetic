import pandas as pd
import numpy as np
import os
import pygame
import random
import sys
import time

# Path setup
output_path = os.path.join('..', 'output')

# Load the dataset with prompts
csv_path = os.path.join(output_path, 'coldplay_with_prompts.csv')

if not os.path.exists(csv_path):
    print(f"Error: Could not find {csv_path}. Please run prompt_generator.py first.")
    exit(1)

df = pd.read_csv(csv_path)

# Get emotion color mapping
emotion_colors = {
    'happy': (255, 221, 0),       # Bright yellow
    'sad': (52, 152, 219),        # Blue
    'energetic': (231, 76, 60),   # Red
    'relaxed': (46, 204, 113),    # Green
    'melancholic': (155, 89, 182), # Purple
    'triumphant': (243, 156, 18),  # Orange
    'ethereal': (26, 188, 156)     # Turquoise
}

# Initialize pygame
pygame.init()

# Set up the window
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Synesthetic Coldplay Demo")

# Set up fonts
pygame.font.init()
title_font = pygame.font.SysFont('Arial', 24)
text_font = pygame.font.SysFont('Arial', 18)

# Particle class
class Particle:
    def __init__(self, emotion, tempo, energy):
        self.x = random.randint(0, width)
        self.y = random.randint(0, height)
        self.size = random.randint(3, 15) * energy
        self.color = emotion_colors.get(emotion, (255, 255, 255))
        self.alpha = random.randint(50, 200)
        self.vx = (random.random() - 0.5) * (tempo / 60)
        self.vy = (random.random() - 0.5) * (tempo / 60)
        
        # Adjust behavior based on emotion
        if emotion == 'happy' or emotion == 'triumphant':
            self.vy -= 1  # Upward tendency
        elif emotion == 'sad' or emotion == 'melancholic':
            self.vy += 0.5  # Downward tendency
        elif emotion == 'energetic':
            self.vx *= 2
            self.vy *= 2
        elif emotion == 'relaxed':
            self.vx *= 0.5
            self.vy *= 0.5
    
    def update(self):
        self.x += self.vx
        self.y += self.vy
        
        # Wrap around screen edges
        if self.x < 0:
            self.x = width
        elif self.x > width:
            self.x = 0
        
        if self.y < 0:
            self.y = height
        elif self.y > height:
            self.y = 0
            
        # Add some randomness
        self.x += random.randint(-1, 1)
        self.y += random.randint(-1, 1)
        
        # Slightly change alpha for pulsing effect
        self.alpha += random.randint(-5, 5)
        self.alpha = max(30, min(200, self.alpha))
    
    def draw(self, surface):
        # Create a surface with per-pixel alpha
        s = pygame.Surface((self.size * 2, self.size * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.color, self.alpha), (self.size, self.size), self.size)
        surface.blit(s, (int(self.x - self.size), int(self.y - self.size)))

# Function to create particles for a song
def create_particles(song, num_particles=100):
    emotion = song['primary_emotion']
    tempo = song['tempo']
    energy = song['energy']
    
    particles = []
    for _ in range(num_particles):
        particles.append(Particle(emotion, tempo, energy))
    
    return particles

# Main loop
def main():
    running = True
    clock = pygame.time.Clock()
    
    # Start with the most popular song
    current_song_index = df['popularity'].idxmax()
    song_cycle_time = time.time()  # Track when to switch songs
    
    current_song = df.iloc[current_song_index]
    particles = create_particles(current_song, num_particles=150)
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    # Change song on spacebar
                    current_song_index = (current_song_index + 1) % len(df)
                    current_song = df.iloc[current_song_index]
                    particles = create_particles(current_song, num_particles=150)
        
        # Auto-change song every 10 seconds
        if time.time() - song_cycle_time > 10:
            current_song_index = (current_song_index + 1) % len(df)
            current_song = df.iloc[current_song_index]
            particles = create_particles(current_song, num_particles=150)
            song_cycle_time = time.time()
        
        # Fill the screen with black
        screen.fill((0, 0, 0))
        
        # Update and draw particles
        for particle in particles:
            particle.update()
            particle.draw(screen)
        
        # Draw song information
        title_surface = title_font.render(current_song['name'], True, (255, 255, 255))
        album_surface = text_font.render(current_song['album_name'], True, (200, 200, 200))
        emotion_surface = text_font.render(f"Emotion: {current_song['primary_emotion']}", True, emotion_colors.get(current_song['primary_emotion'], (255, 255, 255)))
        
        screen.blit(title_surface, (20, 20))
        screen.blit(album_surface, (20, 50))
        screen.blit(emotion_surface, (20, 80))
        
        # Draw instructions
        instructions = text_font.render("Space: Change Song | ESC: Quit", True, (150, 150, 150))
        screen.blit(instructions, (width - instructions.get_width() - 20, height - instructions.get_height() - 20))
        
        # Update the display
        pygame.display.flip()
        
        # Cap the frame rate
        clock.tick(60)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()