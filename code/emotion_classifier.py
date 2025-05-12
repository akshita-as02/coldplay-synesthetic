import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# This class will classify song emotions based on audio features
class EmotionClassifier:
    def __init__(self):
        # Define emotion categories and their characteristic features
        self.emotion_categories = {
            'happy': {'valence': 'high', 'energy': 'high', 'tempo': 'medium-high'},
            'sad': {'valence': 'low', 'energy': 'low', 'tempo': 'low'},
            'energetic': {'valence': 'medium', 'energy': 'high', 'tempo': 'high'},
            'relaxed': {'valence': 'medium-high', 'energy': 'low', 'tempo': 'low'},
            'melancholic': {'valence': 'low', 'energy': 'medium', 'tempo': 'medium-low'},
            'triumphant': {'valence': 'high', 'energy': 'high', 'tempo': 'medium-high'},
            'ethereal': {'valence': 'medium', 'energy': 'low', 'acousticness': 'high'}
        }

    def classify_song(self, song_features):
        """Classify a song's emotion based on its features"""
        # Extract relevant features
        valence = song_features.get('valence', 0.5)
        energy = song_features.get('energy', 0.5)
        tempo = song_features.get('tempo', 0)
        
        # Normalize tempo to 0-1 range (assuming tempo ranges from 60-180 BPM)
        normalized_tempo = min(1.0, max(0.0, (tempo - 60) / 120))
        acousticness = song_features.get('acousticness', 0.3)
        
        # Calculate scores for each emotion based on feature match
        scores = {}
        
        # Happy: high valence, high energy
        scores['happy'] = (valence * 0.6) + (energy * 0.3) + (normalized_tempo * 0.1)
        
        # Sad: low valence, low energy
        scores['sad'] = ((1 - valence) * 0.6) + ((1 - energy) * 0.3) + ((1 - normalized_tempo) * 0.1)
        
        # Energetic: high energy, high tempo
        scores['energetic'] = (energy * 0.5) + (normalized_tempo * 0.4) + (valence * 0.1)
        
        # Relaxed: medium valence, low energy, low tempo
        scores['relaxed'] = ((1 - energy) * 0.5) + ((1 - normalized_tempo) * 0.3) + (valence * 0.2)
        
        # Melancholic: low valence, medium energy
        scores['melancholic'] = ((1 - valence) * 0.6) + (abs(energy - 0.5) * 0.3) + ((1 - normalized_tempo) * 0.1)
        
        # Triumphant: high valence, high energy
        scores['triumphant'] = (valence * 0.4) + (energy * 0.4) + (normalized_tempo * 0.2)
        
        # Ethereal: medium valence, high acousticness, low energy
        scores['ethereal'] = (acousticness * 0.5) + ((1 - energy) * 0.3) + (abs(valence - 0.5) * 0.2)
        
        # Find the emotion with the highest score
        emotion = max(scores, key=scores.get)
        confidence = scores[emotion]
        
        # Return the emotion and the full scores
        return {
            'primary_emotion': emotion,
            'confidence': confidence,
            'scores': scores
        }

    def classify_dataset(self, df):
        """Classify all songs in a dataframe"""
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Apply classification to each row
        classifications = []
        for _, row in df.iterrows():
            features = row.to_dict()
            classification = self.classify_song(features)
            classifications.append(classification)
        
        # Add classification results to the dataframe
        result_df['primary_emotion'] = [c['primary_emotion'] for c in classifications]
        result_df['emotion_confidence'] = [c['confidence'] for c in classifications]
        
        # Add individual emotion scores
        for emotion in self.emotion_categories.keys():
            result_df[f'{emotion}_score'] = [c['scores'][emotion] for c in classifications]
        
        return result_df

    def get_emotion_color_map(self):
        """Return colors associated with emotions for visualization"""
        return {
            'happy': '#FFDD00',      # Bright yellow
            'sad': '#3498DB',        # Blue
            'energetic': '#E74C3C',  # Red
            'relaxed': '#2ECC71',    # Green
            'melancholic': '#9B59B6', # Purple
            'triumphant': '#F39C12',  # Orange
            'ethereal': '#1ABC9C'     # Turquoise
        }


# If the script is run directly, perform a test classification
if __name__ == "__main__":
    # Path setup
    data_path = os.path.join('..', 'data')
    output_path = os.path.join('..', 'output')
    os.makedirs(output_path, exist_ok=True)
    
    # Load the dataset
    csv_path = os.path.join(data_path, 'Coldplay.csv')
    df = pd.read_csv(csv_path)
    
    # Initialize and apply the classifier
    classifier = EmotionClassifier()
    df_with_emotions = classifier.classify_dataset(df)
    
    # Save the results
    df_with_emotions.to_csv(os.path.join(output_path, 'coldplay_with_emotions.csv'), index=False)
    
    # Create a pie chart of emotion distribution
    emotion_counts = df_with_emotions['primary_emotion'].value_counts()
    plt.figure(figsize=(10, 8))
    plt.pie(
        emotion_counts, 
        labels=emotion_counts.index, 
        autopct='%1.1f%%',
        colors=[classifier.get_emotion_color_map()[e] for e in emotion_counts.index],
        startangle=90
    )
    plt.axis('equal')
    plt.title('Distribution of Emotions in Coldplay Songs')
    plt.savefig(os.path.join(output_path, 'emotion_distribution.png'))
    
    # Print summary
    print(f"Classified {len(df)} songs into emotions:")
    for emotion, count in emotion_counts.items():
        print(f"- {emotion}: {count} songs ({count/len(df)*100:.1f}%)")
    
    print(f"\nResults saved to {os.path.join(output_path, 'coldplay_with_emotions.csv')}")
    print(f"Chart saved to {os.path.join(output_path, 'emotion_distribution.png')}")
    
    # Display some example classifications
    print("\nExample classifications:")
    sample_songs = df_with_emotions.sort_values('popularity', ascending=False).head(10)
    for _, song in sample_songs.iterrows():
        print(f"{song['name']} ({song['album_name']}): {song['primary_emotion']} ({song['emotion_confidence']:.2f})")