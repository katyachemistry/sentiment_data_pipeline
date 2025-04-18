import pandas as pd
import numpy as np
import argparse
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling

def load_data(path: str = 'data/cleaned_data.parquet', new_labels_path = 'data/active_learning/uncertain_batch_1_labeled.csv', first_launch=True):
    print("📂 Loading dataset...")
    if first_launch:
        data = pd.read_parquet(path)
        # data = data.dropna(subset=['text'])
        return data
    else: 
        labeled_data = pd.read_csv('data/active_learning/labeled_data.csv', usecols=['text', 'sentiment'])
        new_labels = pd.read_csv(new_labels_path)
        labeled_data = pd.concat([labeled_data, new_labels])
        labeled_data.reset_index(inplace=True, drop=True)
        return labeled_data

def run_active_learning(data: pd.DataFrame, batch_num = 0, n_queries: int = 200, save_model=False):

    if batch_num == 0:
        # Encode sentiment labels (-1, 0, 1)
        data = data[data['text'] != '']
        label_encoder = LabelEncoder()
        labeled_mask = data['sentiment'].notna()
        data.loc[labeled_mask, 'sentiment'] = label_encoder.fit_transform(data.loc[labeled_mask, 'sentiment'])

        labeled_data = data[labeled_mask]
        unlabeled_data = data[~labeled_mask]

    else:
        labeled_data = data.copy()
        unlabeled_data = pd.read_csv('data/active_learning/unlabeled_data.csv')

    labeled_data.dropna(subset=['text'], inplace=True)
    labeled_data.reset_index(drop=True).to_csv('data/active_learning/labeled_data.csv')
    print('Saved labeled data to data/active_learning/labeled_data.csv')

    # TF-IDF vectorization
    print("🧪 Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X_labeled = vectorizer.fit_transform(labeled_data['text'])
    y_labeled = labeled_data['sentiment'].astype(int).values
    X_unlabeled = vectorizer.transform(unlabeled_data['text'])

    # Base model
    base_estimator = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)

    # Active Learner
    learner = ActiveLearner(
        estimator=base_estimator,
        query_strategy=uncertainty_sampling,
        X_training=X_labeled,
        y_training=y_labeled
    )

    print(f"❓ Querying top {n_queries} uncertain samples...")
    query_idx, _ = learner.query(X_unlabeled, n_instances=n_queries)
    uncertain_samples = unlabeled_data.iloc[query_idx].reset_index(drop=True)
    remaining_unlabeled = unlabeled_data.drop(unlabeled_data.index[query_idx]).reset_index(drop=True)
        
    output_path = f'data/active_learning/uncertain_batch_{batch_num}.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    uncertain_samples.to_csv(output_path, index=False)

    remaining_unlabeled.to_csv('data/active_learning/unlabeled_data.csv', index=False)
    print(f"📦 Remaining unlabeled data saved to: data/active_learning/unlabeled_data.csv")

    print(f"✅ Saved {n_queries} most uncertain samples to: {output_path}")

    if save_model:
        import pickle
        with open(f'model_{batch_num}.pkl', 'wb') as file:
            pickle.dump(base_estimator, file)
        print(f"✅ Saved model to model_{batch_num}.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run active learning to select most uncertain samples.")
    parser.add_argument("--input", type=str, default="data/cleaned_data.parquet", help="Path to raw cleaned dataset")
    parser.add_argument("--output", type=str, default="data/uncertain_batch_1.csv", help="Path to save uncertain samples")
    parser.add_argument("--n_queries", type=int, default=200, help="Number of uncertain samples to query")

    args = parser.parse_args()

    run_active_learning(
        input_path=args.input,
        cleaned_path=args.cleaned,
        output_path=args.output,
        n_queries=args.n_queries
    )
