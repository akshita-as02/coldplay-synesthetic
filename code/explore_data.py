import pandas as pd
import matplotlib.pyplot as plt
import os

# Create paths
data_path = os.path.join('..', 'data')  # Path to data folder
output_path = os.path.join('..', 'output')  # Path to output folder

# Create output folder if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Load the Coldplay dataset
csv_path = os.path.join(data_path, 'Coldplay.csv')
df = pd.read_csv(csv_path)

# Print basic information
print(f"Total songs: {len(df)}")
print("\nFirst 5 songs:")
print(df[['name', 'album_name', 'release_date', 'popularity']].head())

print("\nAvailable features:")
print(df.columns.tolist())

# Basic statistics
print("\nBasic statistics:")
print(df[['duration', 'tempo', 'energy', 'valence']].describe())

# Create a simple visualization - Tempo vs Energy
plt.figure(figsize=(10, 6))
plt.scatter(df['tempo'], df['energy'], alpha=0.7)

# Add song labels to some points (just labeling a few popular songs)
top_songs = df.sort_values('popularity', ascending=False).head(10)
for _, song in top_songs.iterrows():
    plt.annotate(song['name'], 
                (song['tempo'], song['energy']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8)

plt.title('Coldplay Songs: Tempo vs Energy')
plt.xlabel('Tempo (BPM)')
plt.ylabel('Energy')
plt.grid(True, alpha=0.3)

# Save the figure
plt.savefig(os.path.join(output_path, 'tempo_vs_energy.png'))
plt.close()

# Create album evolution plot
# First, get mean values per album
album_stats = df.groupby('album_name').agg({
    'name': 'count',
    'tempo': 'mean',
    'energy': 'mean',
    'valence': 'mean',
    'acousticness': 'mean'
}).reset_index()

album_stats = album_stats.rename(columns={'name': 'song_count'})

# Sort albums chronologically (roughly)
album_order = [
    'Parachutes', 'A Rush of Blood to the Head', 'X&Y', 
    'Viva la Vida or Death and All His Friends', 'Mylo Xyloto',
    'Ghost Stories', 'A Head Full of Dreams', 'Everyday Life',
    'Music Of The Spheres'
]

# Filter to only include albums in our order list
album_stats = album_stats[album_stats['album_name'].isin(album_order)]
album_stats['album_name'] = pd.Categorical(
    album_stats['album_name'], 
    categories=album_order, 
    ordered=True
)
album_stats = album_stats.sort_values('album_name')

# Plot evolution
plt.figure(figsize=(14, 10))

plt.subplot(3, 1, 1)
plt.plot(album_stats['album_name'], album_stats['tempo'], 'o-')
plt.title("Tempo Evolution Across Albums")
plt.xticks(rotation=45)
plt.ylabel("Average Tempo (BPM)")

plt.subplot(3, 1, 2)
plt.plot(album_stats['album_name'], album_stats['energy'], 'o-')
plt.title("Energy Evolution Across Albums")
plt.xticks(rotation=45)
plt.ylabel("Average Energy")

plt.subplot(3, 1, 3)
plt.plot(album_stats['album_name'], album_stats['valence'], 'o-')
plt.title("Valence (Positivity) Evolution Across Albums")
plt.xticks(rotation=45)
plt.ylabel("Average Valence")

plt.tight_layout()
plt.savefig(os.path.join(output_path, 'album_evolution.png'))
plt.close()

print(f"\nCharts saved to the 'output' folder")
print("Data exploration complete!")