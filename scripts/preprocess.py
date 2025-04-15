import spacy
import pandas as pd
import re
import string

def process_and_save_data(input_file_path, output_file_path):
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

    # Load dataset
    agg_data = pd.read_csv(input_file_path)
    agg_data.drop_duplicates(inplace=True)

    # Clean text
    agg_data['text'] = agg_data['text'].apply(clean_text)

    # Remove rows with no English letters
    agg_data = agg_data[agg_data['text'].apply(contains_letters)]

    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")

    # Preprocess text with spaCy
    agg_data['text'] = agg_data['text'].apply(lambda text: preprocess_spacy(text, nlp))

    # Save cleaned data to a new CSV file
    agg_data.to_csv(output_file_path, index=False)

    print(f"Data cleaned and saved to {output_file_path}")


if __name__ == "__main__":
    # Define file paths
    input_file = 'data/aggregated_dataset.csv'
    output_file = 'data/cleaned_dataset.csv'

    # Process and save the data
    process_and_save_data(input_file, output_file)
