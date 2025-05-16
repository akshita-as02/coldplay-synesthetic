import os
import pandas as pd
import numpy as np
import pygame
import time
import random
import math
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm

# Path setup
data_path = os.path.join('..', 'data')
output_path = os.path.join('..', 'output')
audio_path = os.path.join(data_path, 'audio')  # Where your MP3 files are stored

# Make sure output directory exists
os.makedirs(output_path, exist_ok=True)

# Load Coldplay dataset with metadata
csv_path = os.path.join(data_path, 'Coldplay.csv')
if not os.path.exists(csv_path):
    print(f"Error: Could not find {csv_path}")
    exit(1)

df = pd.read_csv(csv_path)

# Emotion color mapping
emotion_colors = {
    'happy': (255, 221, 0),       # Bright yellow
    'sad': (52, 152, 219),        # Blue
    'energetic': (231, 76, 60),   # Red
    'relaxed': (46, 204, 113),    # Green
    'melancholic': (155, 89, 182), # Purple
    'triumphant': (243, 156, 18),  # Orange
    'ethereal': (26, 188, 156)     # Turquoise
}

class AudioAnalyzer:
    def __init__(self):
        pass
    
    # def segment_song(self, audio_path, segment_length=1.0):
    #     """Break a song into short segments for analysis"""
    #     print(f"Loading audio: {audio_path}")
    #     y, sr = librosa.load(audio_path)
        
    #     # Calculate number of samples per segment
    #     samples_per_segment = int(segment_length * sr)
        
    #     # Determine number of segments
    #     num_segments = int(np.ceil(len(y) / samples_per_segment))
        
    #     print(f"Dividing into {num_segments} segments...")
    #     segments = []
    #     for i in tqdm(range(num_segments)):
    #         start = i * samples_per_segment
    #         end = min(start + samples_per_segment, len(y))
    #         segment = y[start:end]
            
    #         # Pad last segment if needed
    #         if len(segment) < samples_per_segment:
    #             segment = np.pad(segment, (0, samples_per_segment - len(segment)))
            
    #         segments.append({
    #             'audio': segment,
    #             'start_time': start / sr,
    #             'end_time': end / sr
    #         })
        
    #     # Extract overall tempo for the entire song
    #     tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    #     print(f"Overall song tempo: {tempo:.1f} BPM")
        
    #     # Detect beats for the entire song
    #     _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    #     beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        
    #     return segments, sr, beat_times, tempo

    def segment_song(self, audio_path, segment_length=1.0):
        """Break a song into short segments for analysis"""
        print(f"Loading audio: {audio_path}")
        y, sr = librosa.load(audio_path)
    
        # Calculate number of samples per segment
        samples_per_segment = int(segment_length * sr)
    
        # Determine number of segments
        num_segments = int(np.ceil(len(y) / samples_per_segment))
    
        print(f"Dividing into {num_segments} segments...")
        segments = []
        for i in tqdm(range(num_segments)):
            start = i * samples_per_segment
            end = min(start + samples_per_segment, len(y))
            segment = y[start:end]
            
            # Pad last segment if needed
            if len(segment) < samples_per_segment:
                segment = np.pad(segment, (0, samples_per_segment - len(segment)))
            
            segments.append({
                'audio': segment,
                'start_time': start / sr,
                'end_time': end / sr
            })
        
        # Extract overall tempo for the entire song
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        # Fix: Convert tempo to a scalar if it's an array
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo.item())
        print(f"Overall song tempo: {tempo:.1f} BPM")
        
        # Detect beats for the entire song
        _, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        
        return segments, sr, beat_times, tempo
        
    def extract_segment_features(self, segment, sr):
        """Extract audio features from a segment"""
        audio = segment['audio']
        
        # Energy (volume)
        rms = librosa.feature.rms(y=audio)[0]
        energy = float(np.mean(rms))
        
        # Spectral features (brightness)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        brightness = float(np.mean(spectral_centroid))
        
        # Chromagram (harmonic content)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        
        # Detect if major or minor
        major_profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        minor_profile = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
        
        chroma_sum = np.sum(chroma, axis=1)
        is_major = True
        if np.sum(chroma_sum) > 0:  # Avoid division by zero
            chroma_norm = chroma_sum / np.sum(chroma_sum)
            major_corr = np.correlate(chroma_norm, major_profile)
            minor_corr = np.correlate(chroma_norm, minor_profile)
            is_major = float(major_corr) > float(minor_corr)
        
        # Onset strength (for beat emphasis)
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        
        return {
            'start_time': segment['start_time'],
            'end_time': segment['end_time'],
            'energy': energy,
            'brightness': brightness,
            'is_major': is_major,
            'onset_strength': float(np.mean(onset_env))
        }
    
    def analyze_song(self, audio_file):
        """Analyze a song and extract features for visualization"""
        full_path = os.path.join(audio_path, audio_file)
        if not os.path.exists(full_path):
            print(f"Error: Audio file not found at {full_path}")
            return None
        
        # Get song metadata from CSV
        song_name = os.path.splitext(audio_file)[0]
        song_info = df[df['name'].str.contains(song_name, case=False)]
        
        if len(song_info) == 0:
            print(f"Warning: Could not find metadata for {song_name} in CSV")
            metadata = {
                'name': song_name,
                'album_name': 'Unknown Album',
                'tempo': 120,
                'energy': 0.5,
                'valence': 0.5
            }
        else:
            metadata = song_info.iloc[0].to_dict()
        
        # Segment the song
        segments, sr, beat_times, tempo = self.segment_song(full_path, segment_length=0.5)
        
        # Process each segment
        print("Extracting features from segments...")
        segments_with_features = []
        for segment in tqdm(segments):
            features = self.extract_segment_features(segment, sr)
            segments_with_features.append(features)
        
        # Add beat times and metadata
        result = {
            'metadata': metadata,
            'segments': segments_with_features,
            'beat_times': beat_times.tolist(),
            'tempo': tempo
        }
        
        return result


class EmotionClassifier:
    def __init__(self):
        pass
    
    def classify_segment(self, segment):
        """Classify the emotional content of a segment"""
        # Extract relevant features
        energy = segment['energy']
        is_major = segment['is_major'] 
        brightness = segment['brightness']
        
        # Normalize brightness
        normalized_brightness = min(1.0, max(0.0, brightness / 5000))
        
        # Calculate scores for each emotion
        scores = {}
        
        # Happy: high energy, major key, bright sound
        scores['happy'] = (0.5 if is_major else 0.0) + (energy * 0.3) + (normalized_brightness * 0.2)
        
        # Sad: low energy, minor key, darker sound
        scores['sad'] = (0.5 if not is_major else 0.0) + ((1 - energy) * 0.3) + ((1 - normalized_brightness) * 0.2)
        
        # Energetic: high energy
        scores['energetic'] = (energy * 0.7) + (normalized_brightness * 0.3)
        
        # Relaxed: medium energy, major key
        scores['relaxed'] = (0.3 if is_major else 0.0) + (abs(energy - 0.5) * 0.5) + ((1 - normalized_brightness) * 0.2)
        
        # Melancholic: minor key, medium-low energy
        scores['melancholic'] = (0.4 if not is_major else 0.0) + ((1 - energy) * 0.4) + (abs(normalized_brightness - 0.3) * 0.2)
        
        # Triumphant: major key, high energy
        scores['triumphant'] = (0.5 if is_major else 0.0) + (energy * 0.5)
        
        # Ethereal: medium brightness, medium-low energy
        scores['ethereal'] = (abs(normalized_brightness - 0.5) * 0.5) + (abs(energy - 0.3) * 0.5)
        
        # Find the emotion with the highest score
        emotion = max(scores, key=scores.get)
        confidence = scores[emotion]
        
        return {
            'primary_emotion': emotion,
            'confidence': confidence,
            'scores': scores
        }
    
    def classify_song(self, song_data):
        """Classify emotions for all segments in a song"""
        segments = song_data['segments']
        
        print("Classifying emotions for each segment...")
        for segment in tqdm(segments):
            classification = self.classify_segment(segment)
            segment.update(classification)
        
        return song_data


class VisualGenerator:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Synesthetic Coldplay")
        
        # Initialize fonts
        self.title_font = pygame.font.SysFont('Arial', 28)
        self.info_font = pygame.font.SysFont('Arial', 20)
        
        # Create particle systems for each emotion
        self.particles = []
        self.max_particles = 500
    
    # def create_particles_for_segment(self, segment, beat_pulse=0):
    #     """Create particles based on segment's emotion and features"""
    #     # Clear existing particles if changing emotion dramatically
    #     if len(self.particles) > 0 and random.random() < 0.3:
    #         self.particles = self.particles[:len(self.particles)//2]
        
    #     # Get segment properties
    #     emotion = segment['primary_emotion']
    #     energy = segment['energy'] 
    #     color_base = emotion_colors.get(emotion, (255, 255, 255))
        
    #     # Add beat pulse effect
    #     color = list(color_base)
    #     for i in range(3):
    #         color[i] = min(255, int(color[i] * (1 + beat_pulse * 0.5)))
        
    #     # Calculate how many particles to add
    #     new_particles_count = int(10 + energy * 20 + beat_pulse * 20)
        
    #     # Create new particles
    #     for _ in range(new_particles_count):
    #         if len(self.particles) >= self.max_particles:
    #             # Replace a random particle
    #             idx = random.randint(0, len(self.particles) - 1)
    #             self.particles.pop(idx)
            
    #         # Create particle with properties based on emotion
    #         particle = {
    #             'x': random.randint(0, self.width),
    #             'y': random.randint(0, self.height),
    #             'size': random.randint(5, 20) * (energy + beat_pulse),
    #             'color': color,
    #             'alpha': random.randint(100, 200),
    #             'vx': (random.random() - 0.5) * (2 + energy * 3 + beat_pulse * 2),
    #             'vy': (random.random() - 0.5) * (2 + energy * 3 + beat_pulse * 2),
    #             'life': random.randint(20, 60)
    #         }
            
    #         # Adjust movement based on emotion
    #         if emotion == 'happy' or emotion == 'triumphant':
    #             particle['vy'] -= (1 + beat_pulse)  # Upward tendency
    #         elif emotion == 'sad' or emotion == 'melancholic':
    #             particle['vy'] += (0.5 + beat_pulse * 0.2)  # Downward tendency
    #         elif emotion == 'energetic':
    #             particle['vx'] *= (1.5 + beat_pulse * 0.5)
    #             particle['vy'] *= (1.5 + beat_pulse * 0.5)
    #         elif emotion == 'relaxed':
    #             particle['vx'] *= 0.5
    #             particle['vy'] *= 0.5
            
    #         self.particles.append(particle)
    
    def create_particles_for_segment(self, segment, beat_pulse=0):
        """Create particles based on segment's emotion and features"""
        # Always maintain a minimum number of particles
        min_particles = 50  # Ensure there are always at least this many particles
        
        # Get segment properties
        emotion = segment['primary_emotion']
        energy = max(0.3, segment['energy'])  # Ensure minimum energy level
        color_base = emotion_colors.get(emotion, (255, 255, 255))
        
        # Emotion-specific speed modifiers
        speed_modifiers = {
            'happy': 1.2,      # Slightly faster for happy songs
            'sad': 0.4,        # Much slower for sad songs
            'energetic': 2.0,   # Very fast for energetic songs
            'relaxed': 0.6,     # Slower for relaxed songs
            'melancholic': 0.5, # Slow for melancholic songs
            'triumphant': 1.5,  # Fast for triumphant songs
            'ethereal': 0.8     # Moderate for ethereal songs
        }
        
        # Get the speed modifier for this emotion
        speed_modifier = speed_modifiers.get(emotion, 1.0)
        
        # Add beat pulse effect
        color = list(color_base)
        for i in range(3):
            color[i] = min(255, int(color[i] * (1 + beat_pulse * 0.5)))
        
        # Calculate how many particles to add - ensure more particles
        new_particles_count = int(20 + energy * 40 + beat_pulse * 30)
        
        # Create new particles
        for _ in range(new_particles_count):
            # If we're at the limit, replace older particles
            if len(self.particles) >= self.max_particles:
                # Replace oldest particles first
                self.particles.pop(0)
            
            # Randomize positions across the screen
            # Make patterns based on emotion
            if emotion == 'happy' or emotion == 'triumphant':
                # More particles at the top for uplifting emotions
                x = random.randint(0, self.width)
                y = random.randint(0, self.height * 2//3)
            elif emotion == 'sad' or emotion == 'melancholic':
                # More particles at the bottom for sad emotions
                x = random.randint(0, self.width)
                y = random.randint(self.height // 3, self.height)
            elif emotion == 'energetic':
                # Particles all over for energetic emotions
                x = random.randint(0, self.width)
                y = random.randint(0, self.height)
            else:
                # Default positioning
                x = random.randint(0, self.width)
                y = random.randint(0, self.height)
            
            # Make particles larger and more visible
            particle = {
                'x': x,
                'y': y,
                'size': random.randint(10, 30) * (energy + beat_pulse),  # Larger particles
                'color': color,
                'alpha': random.randint(150, 255),  # Higher alpha for visibility
                'vx': (random.random() - 0.5) * (3 + energy * 5 + beat_pulse * 3) * speed_modifier,  # Apply speed modifier
                'vy': (random.random() - 0.5) * (3 + energy * 5 + beat_pulse * 3) * speed_modifier,  # Apply speed modifier
                'life': random.randint(30, 90)  # Longer life
            }
            
            # Adjust movement based on emotion
            if emotion == 'happy' or emotion == 'triumphant':
                particle['vy'] -= (1.5 + beat_pulse * 2) * speed_modifier  # Stronger upward tendency
            elif emotion == 'sad' or emotion == 'melancholic':
                particle['vy'] += (1 + beat_pulse * 0.5) * speed_modifier  # Stronger downward tendency
            elif emotion == 'energetic':
                particle['vx'] *= (2 + beat_pulse) * speed_modifier
                particle['vy'] *= (2 + beat_pulse) * speed_modifier
            elif emotion == 'relaxed':
                particle['vx'] *= 0.7 * speed_modifier
                particle['vy'] *= 0.7 * speed_modifier
            
            self.particles.append(particle)
        
        # Add some special effect particles on beats
        if beat_pulse > 0.5:
            # Add a burst of particles on strong beats
            burst_count = int(20 * beat_pulse)
            for _ in range(burst_count):
                # Create particles in a circular burst pattern
                angle = random.random() * 2 * 3.14159  # Random angle
                speed = random.random() * 10 * beat_pulse + 5
                
                # Create the particle
                particle = {
                    'x': self.width // 2,  # Center of screen
                    'y': self.height // 2,
                    'size': random.randint(15, 40) * beat_pulse,
                    'color': color,
                    'alpha': 200,
                    'vx': math.cos(angle) * speed,
                    'vy': math.sin(angle) * speed,
                    'life': random.randint(20, 50)
                }
                self.particles.append(particle)
                
                # If over max, remove oldest
                if len(self.particles) > self.max_particles:
                    self.particles.pop(0)
    
    def update_particles(self):
        """Update all particles"""
        i = 0
        while i < len(self.particles):
            p = self.particles[i]
            
            # Update position
            p['x'] += p['vx']
            p['y'] += p['vy']
            
            # Reduce life
            p['life'] -= 1
            
            # Remove if off-screen or expired
            if (p['x'] < -50 or p['x'] > self.width + 50 or 
                p['y'] < -50 or p['y'] > self.height + 50 or
                p['life'] <= 0):
                self.particles.pop(i)
            else:
                # Add some random movement
                p['vx'] += (random.random() - 0.5) * 0.2
                p['vy'] += (random.random() - 0.5) * 0.2
                
                # Fade out as life decreases
                if p['life'] < 20:
                    p['alpha'] = int(p['alpha'] * (p['life'] / 20))
                
                i += 1
    
    def draw_particles(self):
        """Draw all particles"""
        for p in self.particles:
            # Create a surface with per-pixel alpha
            size = int(p['size'])
            s = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*p['color'], p['alpha']), (size, size), size)
            self.screen.blit(s, (int(p['x'] - size), int(p['y'] - size)))
    
    def draw_song_info(self, song_data, current_segment, current_time):
        """Draw song information overlay"""
        metadata = song_data['metadata']
        
        # Song title
        title_surface = self.title_font.render(metadata['name'], True, (255, 255, 255))
        self.screen.blit(title_surface, (20, 20))
        
        # Album name
        album_surface = self.info_font.render(metadata['album_name'], True, (200, 200, 200))
        self.screen.blit(album_surface, (20, 55))
        
        # Current emotion
        emotion = current_segment['primary_emotion']
        emotion_color = emotion_colors.get(emotion, (255, 255, 255))
        emotion_surface = self.info_font.render(f"Emotion: {emotion}", True, emotion_color)
        self.screen.blit(emotion_surface, (20, 85))
        
        # Time
        minutes, seconds = divmod(int(current_time), 60)
        time_surface = self.info_font.render(f"Time: {minutes}:{seconds:02d}", True, (200, 200, 200))
        self.screen.blit(time_surface, (self.width - time_surface.get_width() - 20, 20))
        
        # Instructions
        instructions = self.info_font.render("ESC: Quit | Space: Pause/Play", True, (150, 150, 150))
        self.screen.blit(instructions, (self.width - instructions.get_width() - 20, self.height - 40))
    
    def visualize_song(self, song_data):
        """Run the visualization for a song"""
        # Load the audio file
        metadata = song_data['metadata']
        song_name = metadata['name']
        
        # Find the audio file
        audio_files = [f for f in os.listdir(audio_path) 
                       if os.path.splitext(f)[0].lower() in song_name.lower() 
                       or song_name.lower() in os.path.splitext(f)[0].lower()]
        
        if not audio_files:
            print(f"Error: Could not find audio file for {song_name}")
            return
        
        audio_file = audio_files[0]
        full_audio_path = os.path.join(audio_path, audio_file)
        
        # Initialize Pygame mixer and load audio
        pygame.mixer.init()
        pygame.mixer.music.load(full_audio_path)
        
        # Set window title
        pygame.display.set_caption(f"Synesthetic Coldplay: {song_name}")
        
        # Get song data
        segments = song_data['segments']
        beat_times = song_data['beat_times']
        
        # Start playback
        pygame.mixer.music.play()
        start_time = time.time()
        paused = False
        clock = pygame.time.Clock()
        
        # Main visualization loop
        running = True
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        if paused:
                            pygame.mixer.music.unpause()
                            start_time = time.time() - current_time
                        else:
                            pygame.mixer.music.pause()
                            current_time = time.time() - start_time
                        paused = not paused
            
            # Get current time in the song
            if not paused:
                current_time = time.time() - start_time
            
            # Find current segment
            current_segment = None
            for segment in segments:
                if segment['start_time'] <= current_time < segment['end_time']:
                    current_segment = segment
                    break
            
            # If at the end of the song, exit
            if current_time > segments[-1]['end_time'] or not pygame.mixer.music.get_busy():
                running = False
                break
            
            # Check if we're on a beat
            beat_pulse = 0
            for beat_time in beat_times:
                # If we're within 0.1 seconds of a beat, consider it "on beat"
                time_since_beat = abs(beat_time - current_time)
                if time_since_beat < 0.1:
                    # Pulse intensity decreases as we get further from the beat
                    beat_pulse = 1.0 - (time_since_beat / 0.1)
                    break
            
            # Create new particles
            if current_segment:
                self.create_particles_for_segment(current_segment, beat_pulse)
            
            # Clear screen
            self.screen.fill((0, 0, 0))
            
            # Update and draw particles
            self.update_particles()
            self.draw_particles()
            
            # Draw song information
            if current_segment:
                self.draw_song_info(song_data, current_segment, current_time)
            
            # Update display
            pygame.display.flip()
            
            # Cap at 60 FPS
            clock.tick(60)
        
        # Clean up
        pygame.mixer.music.stop()


def main():
    # Check if audio folder exists
    if not os.path.exists(audio_path):
        print(f"Error: Audio folder not found at {audio_path}")
        print("Please create a folder called 'audio' in your data directory")
        print("and add some Coldplay MP3 files to it.")
        return
    
    # Get list of audio files
    audio_files = [f for f in os.listdir(audio_path) if f.endswith(('.mp3', '.wav'))]
    
    if not audio_files:
        print("No audio files found. Please add some MP3 files to the audio folder.")
        return
    
    print("Available audio files:")
    for i, file in enumerate(audio_files):
        print(f"{i+1}. {file}")
    
    try:
        selection = int(input("\nSelect a song to visualize (enter number): ")) - 1
        if selection < 0 or selection >= len(audio_files):
            print("Invalid selection. Using the first song.")
            selection = 0
    except:
        print("Invalid input. Using the first song.")
        selection = 0
    
    selected_file = audio_files[selection]
    print(f"\nAnalyzing {selected_file}...")
    
    # Create analyzer and process the song
    analyzer = AudioAnalyzer()
    song_data = analyzer.analyze_song(selected_file)
    
    if not song_data:
        print("Error analyzing song. Exiting.")
        return
    
    # Classify emotions
    classifier = EmotionClassifier()
    song_data_with_emotions = classifier.classify_song(song_data)
    
    # Run visualization
    print("\nStarting visualization...")
    visualizer = VisualGenerator()
    visualizer.visualize_song(song_data_with_emotions)
    
    print("Visualization complete.")


if __name__ == "__main__":
    main()