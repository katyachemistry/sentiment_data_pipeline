import pandas as pd

def aggregate_datasets():
    print("Aggregating data...")

    # Load YouTube dataset from Parquet
    youtube_dataset = pd.read_parquet('data/raw/youtube_comments_df.parquet', columns=['comment'])
    youtube_dataset.drop_duplicates('comment', inplace=True)

    print('Leaving YouTube comments with less than 50 words only...')
    youtube_dataset = youtube_dataset[youtube_dataset['comment'].astype(str).apply(lambda x: len(x.split())) < 50]

    youtube_dataset.columns = ['text']
    youtube_dataset['sentiment'] = float('NaN')

    kaggle_dataset = pd.read_parquet('data/raw/sentiment_kaggle.parquet')
    kaggle_dataset.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']
    kaggle_dataset = kaggle_dataset[['text', 'sentiment']]

    kaggle_dataset['sentiment'] = kaggle_dataset['sentiment'].map({0: 0, 4: 1})

    print("Downsampling Kaggle dataset to match YouTube dataset size...")
    kaggle_dataset = kaggle_dataset.sample(60000, random_state=42)

    # Concatenate datasets
    concatenated = pd.concat([kaggle_dataset, youtube_dataset], ignore_index=True)
    concatenated = concatenated.sample(frac=1, random_state=42).reset_index(drop=True)
    concatenated['sentiment'] = float('NaN')

    print("Datasets concatenated successfully. Here's a preview:")

    # Save as Parquet
    print("Saving the concatenated dataset to 'aggregated_dataset.parquet'...")
    concatenated.to_parquet('data/aggregated_dataset.parquet', index=False)

    print("✅ Process completed successfully!")

if __name__ == "__main__":
    aggregate_datasets()
