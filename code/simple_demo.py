import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import to_rgba
import random

# Path setup
output_path = os.path.join('..', 'output')
demo_path = os.path.join(output_path, 'demo')
os.makedirs(demo_path, exist_ok=True)

# Load the dataset with prompts
csv_path = os.path.join(output_path, 'coldplay_with_prompts.csv')

if not os.path.exists(csv_path):
    print(f"Error: Could not find {csv_path}. Please run prompt_generator.py first.")
    exit(1)

df = pd.read_csv(csv_path)

# Get the most popular song
top_song = df.sort_values('popularity', ascending=False).iloc[0]
print(f"Creating demo for: {top_song['name']} ({top_song['album_name']})")
print(f"Emotion: {top_song['primary_emotion']}")
print(f"Prompt: {top_song['visual_prompt']}")

# Get emotion color mapping
emotion_colors = {
    'happy': '#FFDD00',      # Bright yellow
    'sad': '#3498DB',        # Blue
    'energetic': '#E74C3C',  # Red
    'relaxed': '#2ECC71',    # Green
    'melancholic': '#9B59B6', # Purple
    'triumphant': '#F39C12',  # Orange
    'ethereal': '#1ABC9C'     # Turquoise
}

# Get color for this song's emotion
main_color = emotion_colors.get(top_song['primary_emotion'], '#FFFFFF')
main_color_rgba = to_rgba(main_color, alpha=0.7)

# Create a simple particle system as a visual example
# This is a very basic simulation - in reality you would use more sophisticated visuals
class ParticleSystem:
    def __init__(self, emotion, tempo, energy, valence):
        self.emotion = emotion
        self.tempo = tempo
        self.energy = energy
        self.valence = valence
        self.color = emotion_colors.get(emotion, '#FFFFFF')
        
        # Create particles
        self.num_particles = int(100 + energy * 200)  # More particles for high energy
        self.positions = np.random.rand(self.num_particles, 2)  # x, y positions
        self.velocities = (np.random.rand(self.num_particles, 2) - 0.5) * (tempo / 120)  # Velocity affected by tempo
        self.sizes = np.random.rand(self.num_particles) * energy * 150 + 10  # Size affected by energy
        
        # Different movement patterns based on emotion
        if emotion == 'happy' or emotion == 'triumphant':
            # Upward movement
            self.velocities[:, 1] += 0.01
        elif emotion == 'sad' or emotion == 'melancholic':
            # Downward movement
            self.velocities[:, 1] -= 0.01
        elif emotion == 'energetic':
            # Faster, more chaotic movement
            self.velocities *= 1.5
        elif emotion == 'relaxed':
            # Slower, gentler movement
            self.velocities *= 0.5
        elif emotion == 'ethereal':
            # Circular movement
            self.velocities = np.array([
                np.cos(np.linspace(0, 2*np.pi, self.num_particles)),
                np.sin(np.linspace(0, 2*np.pi, self.num_particles))
            ]).T * 0.01
    
    def update(self):
        # Update positions
        self.positions += self.velocities * 0.1
        
        # Wrap particles that go off-screen
        self.positions = self.positions % 1.0
        
        # Add some randomness
        self.positions += (np.random.rand(self.num_particles, 2) - 0.5) * 0.01
        
        return self.positions, self.sizes

# Create a particle system based on the song
particle_system = ParticleSystem(
    top_song['primary_emotion'],
    top_song['tempo'],
    top_song['energy'],
    top_song['valence']
)

# Set up the figure for animation
fig, ax = plt.figure(figsize=(10, 8)), plt.subplot(111)
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# Add song title
ax.text(0.5, 0.95, top_song['name'], 
        color='white', fontsize=16, ha='center', va='top')
ax.text(0.5, 0.90, f"{top_song['album_name']} - {top_song['primary_emotion']}",
        color='white', fontsize=12, ha='center', va='top', alpha=0.7)

# Create the scatter plot
scatter = ax.scatter([], [], s=[], c=main_color_rgba, alpha=0.7)

# Update function for animation
def update(frame):
    positions, sizes = particle_system.update()
    scatter.set_offsets(positions)
    scatter.set_sizes(sizes)
    
    # Add a color variation based on frame
    scatter.set_color(to_rgba(main_color, alpha=0.5 + 0.2 * np.sin(frame * 0.1)))
    
    return scatter,

# Create the animation
animation = FuncAnimation(fig, update, frames=100, interval=50, blit=True)

# Save the animation
animation.save(os.path.join(demo_path, f"{top_song['name'].replace(' ', '_')}_demo.gif"), 
               dpi=80, writer='pillow')

print(f"Demo animation saved to: {os.path.join(demo_path, f'{top_song['name'].replace(' ', '_')}_demo.gif')}")