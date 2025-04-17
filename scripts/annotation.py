import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import IsolationForest
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import entropy


def detect_anomalies(input_path: str, output_path: str = 'data/no_anomaly_data.parquet', sample_size: int = 20000) -> str:
    print("📌 Loading data...")
    data = pd.read_parquet(input_path).sample(sample_size, random_state=42)

    print("🔍 Running anomaly detection...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(data['text'].tolist(), show_progress_bar=True)

    clf = IsolationForest(contamination=0.02, random_state=42)
    preds = clf.fit_predict(embeddings)

    data_clean = data[preds == 1].reset_index(drop=True)
    data_clean.to_parquet(output_path, index=False)
    print(f"✅ Saved cleaned data without anomalies to {output_path} ({len(data_clean)} rows)")

    return output_path


def auto_annotate(data_path: str = 'data/no_anomaly_data.parquet', save_needed: bool = False, sample_size: int = 2000, output_path: str = 'data/auto_annotated.parquet') -> pd.DataFrame:
    print("✍️ Annotating sentiment...")
    data_sample = pd.read_parquet(data_path)
    data_sample = data_sample.sample(sample_size, random_state=42).copy()

    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    results = data_sample['text'].apply(lambda x: classifier(x)[0])

    label_map = {
        "LABEL_0": "-1",  # negative
        "LABEL_1": "0",   # neutral
        "LABEL_2": "1"    # positive
    }

    data_sample['sentiment'] = [label_map[result['label']] for result in results]
    data_sample['confidence'] = [result['score'] for result in results]

    if save_needed:
        data_sample.to_parquet(output_path, index=False)
        print(f"✅ Saved automatically annotated data to {output_path}")

    return data_sample


def analyze_complexity(df: pd.DataFrame, conclusion_txt: str = "complexity_report.txt"):
    print("📊 Starting complexity analysis...")

    os.makedirs("complexity_analysis", exist_ok=True)

    # Ensure correct data types
    df['sentiment'] = df['sentiment'].astype(str)
    df['confidence'] = df['confidence'].astype(float)
    df['text_length'] = df['text'].astype(str).apply(lambda x: len(x.split()))

    # 1. Shannon Entropy (label distribution)
    label_counts = df['sentiment'].value_counts(normalize=True)
    shannon_entropy = entropy(label_counts, base=2)

    plt.figure(figsize=(6, 4))
    sns.barplot(x=label_counts.index, y=label_counts.values)
    plt.title("Sentiment Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Proportion")
    plt.savefig("complexity_analysis/label_distribution.png")
    plt.close()

    # 2. Mutual Information (text length ↔ sentiment)
    X = df[['text_length']]
    y = df['sentiment']
    mi = mutual_info_classif(X, y, discrete_features=True)
    mutual_info_value = mi[0]

    plt.figure(figsize=(6, 4))
    sns.boxplot(x='sentiment', y='text_length', data=df)
    plt.title("Text Length by Sentiment")
    plt.savefig("complexity_analysis/text_length_vs_sentiment.png")
    plt.close()

    # 3. Uncertainty Sampling (confidence distribution)
    plt.figure(figsize=(6, 4))
    sns.histplot(df['confidence'], bins=30, kde=True)
    plt.title("Confidence Distribution")
    plt.xlabel("Confidence Score")
    plt.ylabel("Frequency")
    plt.savefig("complexity_analysis/confidence_distribution.png")
    plt.close()

    # Save confident vs non-confident splits
    threshold = 0.6
    confident = df[df['confidence'] >= threshold]
    non_confident = df[df['confidence'] < threshold]

    confident[['text', 'sentiment']].to_csv("data/auto_annotated_confident.csv", index=False)
    non_confident[['text']].to_csv("data/auto_annotated_non_confident.csv", index=False)

    uncertain_percent = (len(non_confident) / len(df)) * 100

    # Save textual conclusion
    conclusion_path = os.path.join("complexity_analysis", conclusion_txt)
    with open(conclusion_path, "w", encoding="utf-8") as f:
        f.write("🧠 Complexity Analysis Report\n")
        f.write("="*35 + "\n")
        f.write(f"📌 Shannon Entropy of Labels: {shannon_entropy:.4f}\n")
        f.write(f"📌 Mutual Information (Text Length ↔ Sentiment): {mutual_info_value:.4f}\n")
        f.write(f"📌 Percentage of Uncertain Predictions (confidence < {threshold}): {uncertain_percent:.2f}%\n")
        f.write(f"\n✅ Saved {len(confident)} confident → data/auto_annotated_confident.parquet\n")
        f.write(f"⚠️ Saved {len(non_confident)} uncertain → data/auto_annotated_non_confident.parquet\n")
        f.write("\n📊 Visualizations saved in 'complexity_analysis/' folder.\n")

    print("✅ Complexity report and data splits saved.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full sentiment pipeline: anomaly detection, annotation, complexity analysis.")
    parser.add_argument('--input_path', type=str, required=True, help='Path to input parquet file')
    parser.add_argument('--report_name', type=str, default='complexity_report.txt', help='Filename for complexity report')
    args = parser.parse_args()

    print("🚀 Starting processing pipeline...")

    clean_data = detect_anomalies(args.input_path)
    annotated_data = auto_annotate(clean_data)

    analyze_complexity(annotated_data, args.report_name)

    print("🎉 All steps completed successfully!")
