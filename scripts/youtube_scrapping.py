import json
import random
import pandas as pd
import os
import argparse
from tqdm import tqdm
from itertools import islice
from pathlib import Path
import scrapetube
from youtube_comment_downloader import *

def parse_youtube_comments( 
    topics_path='scripts/topics.json',
    output_df_path='data/raw/youtube_comments_df.parquet',
    X=70,
    Y=10,
    K=140,
    seed=42,
    force=False 
):
    assert K % 2 == 0, "⚠️ K must be divisible by 2"

    output_file = Path(output_df_path)
    if output_file.exists() and not force:
        print(f"⚠️ File '{output_df_path}' already exists. Skipping scraping.\n"
              f"✅ Use `force=True` to re-scrape and overwrite the file. It will take ~1.5 hours.")
        return pd.read_parquet(output_df_path)

    # Set random seed for reproducibility
    random.seed(seed)

    # Load topics
    with open(topics_path) as f:
        topics = json.load(f)

    selected_topics = random.sample(topics, min(X, len(topics)))

    downloader = YoutubeCommentDownloader()
    comments_data = {}

    print(f"🔍 Scraping {Y} videos and {K} comments for each of {X} topics...")
    for topic in tqdm(selected_topics, desc="Topics"):
        videos = scrapetube.get_search(topic, limit=Y)
        comments_data[topic] = {}

        for video in tqdm(videos, desc=f"Videos for '{topic}'", leave=False):
            video_id = video['videoId']
            video_url = f'https://www.youtube.com/watch?v={video_id}'

            comms = []

            # Top-K/2 recent comments
            recent_comments = downloader.get_comments_from_url(video_url, sort_by=SORT_BY_RECENT, language='en')
            comms.extend(islice(recent_comments, K // 2))

            # Top-K/2 popular comments
            popular_comments = downloader.get_comments_from_url(video_url, sort_by=SORT_BY_POPULAR, language='en')
            comms.extend(islice(popular_comments, K // 2))

            comments_data[topic][video_id] = list(comms)

    # Flatten the nested dict to rows
    flattened_comments = []
    for topic, videos in tqdm(comments_data.items(), desc="Flattening comments"):
        for video_id, comments in videos.items():
            for comment in comments:
                flattened_comments.append({
                    'topic': topic,
                    'video_id': video_id,
                    'comment': comment.get('text'),
                    'time': comment.get('time'),
                    'author': comment.get('author'),
                    'channel': comment.get('channel'),
                    'votes': comment.get('votes'),
                    'replies': comment.get('replies'),
                    'heart': comment.get('heart'),
                    'reply': comment.get('reply'),
                })

    df = pd.DataFrame(flattened_comments)

    # Optional: convert votes/replies to numeric
    df['votes'] = pd.to_numeric(df['votes'], errors='coerce')
    df['replies'] = pd.to_numeric(df['replies'], errors='coerce')

    Path(os.path.dirname(output_df_path)).mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_df_path, index=False)
    print(f"✅ Saved DataFrame with {len(df)} comments to {output_df_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape YouTube comments for topic-based analysis")
    parser.add_argument('--topics_path', type=str, default='scripts/topics.json', help='Path to topics JSON file')
    parser.add_argument('--output_df_path', type=str, default='data/raw/youtube_comments_df.parquet', help='Output Parquet file path')
    parser.add_argument('--X', type=int, default=20, help='Number of topics to sample')
    parser.add_argument('--Y', type=int, default=50, help='Number of videos per topic')
    parser.add_argument('--K', type=int, default=70, help='Number of comments per video (must be divisible by 2)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--force', action='store_true', help='Force re-scraping even if the output file already exists')

    args = parser.parse_args()
    
    parse_youtube_comments(
        topics_path=args.topics_path,
        output_df_path=args.output_df_path,
        X=args.X,
        Y=args.Y,
        K=args.K,
        seed=args.seed,
        force=args.force
    )
