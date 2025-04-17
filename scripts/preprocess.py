import spacy
import pandas as pd
import re
import string

def load_data(no_anomaly_path: str, auto_annotated_path: str, manual_path: str) -> pd.DataFrame:
    """
    Load and annotate the no_anomaly_data DataFrame using auto and manual annotations.

    Parameters:
        no_anomaly_path (str): Path to the no_anomaly_data parquet file.
        auto_annotated_path (str): Path to the auto_annotated CSV file.
        manual_path (str): Path to the manual annotations CSV file.

    Returns:
        pd.DataFrame: Annotated DataFrame with 'sentiment' column added.
    """
    # Load datasets
    no_anomaly_data = pd.read_parquet(no_anomaly_path)
    auto_annotated = pd.read_csv(auto_annotated_path)
    manual = pd.read_csv(manual_path)

    # Prepare manual data
    manual = manual[['text', 'sentiment']]
    sentiment_mapping = {'Neutral': 0, "Negative": -1, "Positive": 1}
    manual['sentiment'] = manual['sentiment'].apply(lambda x: sentiment_mapping[x])

    # Build sentiment dictionary
    sentiment_dict = {row['text']: row['sentiment'] for _, row in auto_annotated.iterrows()}
    sentiment_dict.update({row['text']: row['sentiment'] for _, row in manual.iterrows()})  # manual overrides auto

    # Map sentiments to the no_anomaly_data
    no_anomaly_data['sentiment'] = no_anomaly_data['text'].map(sentiment_dict)

    return no_anomaly_data


def process_and_save_data(input_data):
    """
    Loads the dataset, cleans the text, processes it with SpaCy,
    and saves the cleaned data to a new CSV file.

    Parameters:
    - input_file_path: The file path of the input CSV.
    - output_file_path: The file path where the cleaned data will be saved.
    """
    def clean_text(text):
        """Cleans the text by removing links, mentions, time formats, numbers, punctuation, and non-ASCII characters."""
        # 1. Remove links
        text = re.sub(r'https?://\S+|www\.\S+', '', text)

        # 2. Remove @mentions / nicknames
        text = re.sub(r'@\w+', '', text)

        # 3. Remove time formats like 1:23 or 12:00
        text = re.sub(r'\b\d{1,2}:\d{2}\b', '', text)

        # 4. Remove all numbers
        text = re.sub(r'\d+', '', text)

        # 5. Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # 6. Remove emojis and other non-ASCII characters
        text = re.sub(r'[^\x00-\x7F]+', '', text)

        return text.strip()

    def contains_letters(text):
        """Checks if the text contains at least one English alphabetic character."""
        return bool(re.search(r'[a-zA-Z]', text))

    def preprocess_spacy(text, nlp):
        """Preprocesses the text using spaCy: tokenization, removing stop words, and lemmatization."""
        doc = nlp(text)
        return " ".join([token.lemma_ for token in doc if not token.is_stop])

    # Initial size
    print(f"Initial data size: {input_data.shape}")

    # Remove duplicates
    input_data.drop_duplicates(inplace=True)
    print(f"After dropping duplicates: {input_data.shape}")

    # Clean text
    input_data['text'] = input_data['text'].apply(clean_text)
    print("Text cleaned")

    # Remove rows with no English letters
    before_filter = input_data.shape[0]
    input_data = input_data[input_data['text'].apply(contains_letters)]
    after_filter = input_data.shape[0]
    print(f"Removed {before_filter - after_filter} rows without English letters. Remaining: {after_filter}")

    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")
    print("spaCy model loaded")

    # Preprocess text with spaCy
    input_data['text'] = input_data['text'].apply(lambda text: preprocess_spacy(text, nlp))
    print(f"Text preprocessing complete. Final data size: {input_data.shape}")
    print(f"Final labelled data size: {input_data[~input_data['sentiment'].isna()].shape}")
    input_data.to_parquet('data/cleaned_data.parquet')

    return input_data

    