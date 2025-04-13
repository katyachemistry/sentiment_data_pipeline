import pandas as pd

def aggregate_datasets():
    print("Aggregating data...")
    
    youtube_dataset = pd.read_csv('data/raw/youtube_comments_df.csv', usecols=['comment'])
    youtube_dataset.drop_duplicates('comment', inplace=True)
    
    print('Leaving youtube comments with less than 50 words only...')
    youtube_dataset = youtube_dataset[youtube_dataset['comment'].astype(str).apply(lambda x: len(x.split())) < 50]
    
    youtube_dataset.columns = ['text']
    youtube_dataset['sentiment'] = float('NaN')
    
    column_names = ['sentiment', 'id', 'date', 'query', 'user', 'text']
    kaggle_dataset = pd.read_csv('data/raw/sentiment_kaggle/training.1600000.processed.noemoticon.csv', 
                                 encoding='latin-1', names=column_names)
    
    kaggle_dataset['sentiment'] = kaggle_dataset['sentiment'].map({0: 0, 4: 1})
    
    print("Downsampling Kaggle dataset to match YouTube dataset size...")
    kaggle_dataset = kaggle_dataset.sample(60000, random_state=42)
    
    kaggle_dataset = kaggle_dataset[['text', 'sentiment']]
    
    concatenated = pd.concat([kaggle_dataset, youtube_dataset], ignore_index=True)
    concatenated = concatenated.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("Datasets concatenated successfully. Here's a preview:")
    print(concatenated.head())
    
    print("Saving the concatenated dataset to 'aggregated_dataset.csv'...")
    concatenated.to_csv('data/aggregated_dataset.csv', index=False)
    
    print("Process completed successfully!")
    return concatenated

if __name__ == "__main__":
    aggregate_datasets()