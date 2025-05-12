import pandas as pd
import numpy as np
import os
import json
import random
from tqdm import tqdm

class PromptGenerator:
    def __init__(self):
        # Create visual mapping for each emotion
        self.emotion_visual_mapping = {
            'happy': {
                'color_palette': ['yellow', 'gold', 'orange'],
                'movement': 'upward floating movement',
                'elements': ['sunshine', 'bubbles', 'stars', 'flowers'],
                'atmosphere': 'bright and joyful',
                'texture': 'soft and warm'
            },
            'sad': {
                'color_palette': ['blue', 'indigo', 'purple'],
                'movement': 'slow downward drifting',
                'elements': ['rain', 'water', 'shadows', 'silhouettes'],
                'atmosphere': 'melancholic and gentle',
                'texture': 'smooth and flowing'
            },
            'energetic': {
                'color_palette': ['red', 'orange', 'pink'],
                'movement': 'rapid pulsing movement',
                'elements': ['fire', 'lightning', 'sparks', 'waves'],
                'atmosphere': 'vibrant and dynamic',
                'texture': 'sharp and electric'
            },
            'relaxed': {
                'color_palette': ['green', 'teal', 'light blue'],
                'movement': 'gentle swaying',
                'elements': ['leaves', 'water', 'clouds', 'fields'],
                'atmosphere': 'peaceful and calm',
                'texture': 'soft and flowing'
            },
            'melancholic': {
                'color_palette': ['deep blue', 'gray', 'purple'],
                'movement': 'slow sweeping motions',
                'elements': ['mist', 'rain', 'empty spaces', 'shadows'],
                'atmosphere': 'thoughtful and introspective',
                'texture': 'smooth and ethereal'
            },
            'triumphant': {
                'color_palette': ['gold', 'orange', 'white'],
                'movement': 'rising motion',
                'elements': ['sunbursts', 'rays of light', 'mountains', 'skies'],
                'atmosphere': 'powerful and uplifting',
                'texture': 'bold and dramatic'
            },
            'ethereal': {
                'color_palette': ['turquoise', 'silver', 'light purple'],
                'movement': 'floating and dreamy',
                'elements': ['stars', 'cosmos', 'fog', 'light rays'],
                'atmosphere': 'dreamy and otherworldly',
                'texture': 'transparent and glowing'
            }
        }
        
        # Add scene templates for each emotion
        self.scene_templates = {
            'happy': [
                "a bright meadow with wildflowers",
                "sunlight streaming through colorful clouds",
                "a celebration with colorful confetti",
                "a summer beach scene with perfect waves"
            ],
            'sad': [
                "a solitary figure on a rainy street",
                "a lone tree in a misty field",
                "raindrops on a window with blurred city lights beyond",
                "an empty room with soft blue light"
            ],
            'energetic': [
                "an electric light show with laser beams",
                "a vibrant dance floor with colorful lights",
                "an exploding firework display",
                "waves crashing against rocks with spray"
            ],
            'relaxed': [
                "a peaceful forest with dappled sunlight",
                "a gentle stream flowing through a green meadow",
                "clouds floating in a clear blue sky",
                "a hammock swaying under palm trees"
            ],
            'melancholic': [
                "a foggy path through autumn woods",
                "rain-soaked streets at twilight",
                "a person looking out at a gray horizon",
                "faded photographs and memories"
            ],
            'triumphant': [
                "a sunrise over mountain peaks",
                "a figure standing victorious on a summit",
                "rays of light breaking through storm clouds",
                "a grand royal palace with golden light"
            ],
            'ethereal': [
                "a cosmic nebula with swirling colors",
                "light filtering through crystal formations",
                "glowing jellyfish in deep waters",
                "aurora borealis dancing across the night sky"
            ]
        }
    
    def generate_prompt_for_song(self, song_info, tempo_modifier=1.0):
        """Generate a visual prompt based on song information"""
        # Extract features
        primary_emotion = song_info['primary_emotion']
        energy = song_info.get('energy', 0.5)
        tempo = song_info.get('tempo', 120) * tempo_modifier
        valence = song_info.get('valence', 0.5)
        
        # Get relevant mappings
        mapping = self.emotion_visual_mapping.get(primary_emotion, self.emotion_visual_mapping['happy'])
        
        # Select a random scene template for this emotion
        scene = random.choice(self.scene_templates.get(primary_emotion, self.scene_templates['happy']))
        
        # Select random elements from the mapping
        color = random.choice(mapping['color_palette'])
        element = random.choice(mapping['elements'])
        
        # Adjust movement based on tempo
        movement = mapping['movement']
        if tempo > 140:
            movement = movement.replace('slow', 'rapid').replace('gentle', 'dynamic')
        elif tempo < 80:
            movement = movement.replace('rapid', 'slow').replace('dynamic', 'gentle')
        
        # Adjust intensity based on energy
        intensity = "vibrant and intense" if energy > 0.7 else "soft and subtle"
        if 0.3 <= energy <= 0.7:
            intensity = "balanced and harmonious"
        
        # Create the prompt
        prompt = f"{scene} with {color} tones, featuring {element}, {movement}, {mapping['atmosphere']} atmosphere, {intensity}, {mapping['texture']} textures, high quality, detailed, cinematic lighting"
        
        return prompt

    def generate_prompts_for_dataset(self, df):
        """Generate prompts for each song in the dataset"""
        # Create a copy of the dataframe
        result_df = df.copy()
        
        # Generate prompts for each song
        prompts = []
        for _, song in tqdm(df.iterrows(), total=len(df), desc="Generating prompts"):
            prompt = self.generate_prompt_for_song(song)
            prompts.append(prompt)
        
        # Add prompts to the dataframe
        result_df['visual_prompt'] = prompts
        
        return result_df


# If the script is run directly, generate prompts for the dataset
if __name__ == "__main__":
    # Path setup
    output_path = os.path.join('..', 'output')
    os.makedirs(output_path, exist_ok=True)
    
    # Load the dataset with emotions
    csv_path = os.path.join(output_path, 'coldplay_with_emotions.csv')
    
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}. Please run emotion_classifier.py first.")
        exit(1)
    
    df = pd.read_csv(csv_path)
    
    # Initialize prompt generator
    generator = PromptGenerator()
    
    # Generate prompts
    df_with_prompts = generator.generate_prompts_for_dataset(df)
    
    # Save the results
    df_with_prompts.to_csv(os.path.join(output_path, 'coldplay_with_prompts.csv'), index=False)
    
    # Display some example prompts
    print("\nExample prompts for popular songs:")
    sample_songs = df_with_prompts.sort_values('popularity', ascending=False).head(5)
    for _, song in sample_songs.iterrows():
        print(f"\n{song['name']} ({song['album_name']})")
        print(f"Emotion: {song['primary_emotion']}")
        print(f"Prompt: {song['visual_prompt']}")
    
    print(f"\nGenerated prompts for {len(df)} songs")
    print(f"Results saved to {os.path.join(output_path, 'coldplay_with_prompts.csv')}")