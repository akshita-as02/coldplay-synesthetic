# coldplay-synesthetic
# Synesthetic Coldplay: A Visual Music Experience

## Project Overview

This is an academic project that aims to explore innovative ways of experiencing music, particularly for individuals with hearing impairments. The core idea is to translate the auditory elements of Coldplay's music into a dynamic and emotionally resonant visual display, creating a form of "synesthetic" experience where sound is perceived through color, movement, and patterns.

By converting musical features such as tempo, beat, energy, and emotional tone into visual cues, this project seeks to make the richness of music more accessible and engaging for a wider audience, including those who may not perceive sound in the traditional sense.

## Core Features

*   **Advanced Audio Analysis**: Leverages the `librosa` library to extract key audio features from Coldplay songs, including:
    *   Tempo and beat timings for rhythmic synchronization.
    *   Energy (RMS) to represent volume and intensity.
    *   Spectral features like brightness.
    *   Harmonic content (chroma features) to determine if segments are in a major or minor key.
*   **Emotion-Driven Visualization**:
    *   Classifies the emotional content of song segments (e.g., happy, sad, energetic, relaxed, melancholic, triumphant, ethereal).
    *   Maps these emotions to distinct color palettes (e.g., yellow for happy, blue for sad).
*   **Dynamic Particle System**:
    *   Utilizes `pygame` to generate a real-time particle animation.
    *   Particle characteristics (color, size, speed, movement patterns, and lifespan) are dynamically modulated by:
        *   The classified **emotion** of the current song segment.
        *   The **energy** level of the music.
        *   The overall **tempo** of the song.
        *   **Beat pulses**, causing particles to react visually to rhythmic accents.
    *   The animation speed is adjusted based on the detected emotion (e.g., slower, more gentle movements for sad or melancholic songs, and faster, more vibrant movements for energetic or happy songs).
*   **Beat Synchronization**: Visual elements, particularly particle generation and movement, are synchronized with the detected beats of the song, creating a cohesive audio-visual experience.
*   **Interactive Controls**:
    *   `ESC` key: Quit the visualization.
    *   `Space` key: Pause or resume the music and visualization.

## Dataset and Audio Files

*   The project uses a CSV file (`Coldplay.csv`, expected in the `data/` directory) containing metadata about Coldplay songs. This metadata is used to supplement the audio analysis.
*   The primary audio processing is done on actual MP3 or WAV audio files that you provide.
*   The script also generates an enhanced CSV (`output/coldplay_with_emotions.csv`) after processing, which includes the classified emotions for song segments. *(This file is generated if `emotion_classifier.py` is run, or implicitly by `beat_sync_demo.py` as it uses the `EmotionClassifier` class).*

## Getting Started

### Prerequisites

Ensure you have Python installed (version 3.7+ recommended). You will also need to install the following Python libraries:

```bash
pip install pandas numpy pygame librosa matplotlib tqdm
```

### Directory Structure

Organize your project files as follows:
Synesthetic_Coldplay/
├── code/

│ └── beat_sync_demo.py

├── data/

│ ├── Coldplay.csv

│ └── audio/

│ └── (Place your Coldplay .mp3 or .wav files here)

└── output/

└── (This directory will be created if it doesn't exist, for output files like coldplay_with_emotions.csv)


### Running the Visualization

1.  **Place Audio Files**: Copy your Coldplay MP3 or WAV files into the `data/audio/` directory.
2.  **Navigate to Code Directory**: Open your terminal or command prompt and navigate to the `code/` directory within your project.
    ```bash
    cd path/to/Synesthetic_Coldplay/code/
    ```
3.  **Run the Script**: Execute the main Python script.
    ```bash
    python beat_sync_demo.py
    ```
4.  **Select a Song**: The script will list the audio files found in the `data/audio/` directory and prompt you to select a song by number.
5.  **Enjoy**: The visualization will start, translating the selected song into a dynamic visual display.

## Project Aim

The primary goal of this academic endeavor is to investigate and demonstrate how musical elements can be effectively translated into a compelling visual narrative. It serves as a proof-of-concept for creating synesthetic music experiences, with a particular focus on enhancing accessibility for individuals with hearing impairments by offering an alternative, visually rich way to connect with music.

## Disclaimer

This project is developed for academic and experimental purposes only. The use of copyrighted music (Coldplay songs) is solely for the purpose of research and demonstration within this non-commercial project. There is no intention to infringe upon any music copyrights. Users are expected to use their own legally obtained audio files for experimentation. The developers do not claim ownership of any copyrighted music used and are not responsible for the misuse of this software.
---
