"""
Dataset loading and feature extraction
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


def load_data(filepath='data/fora_test.csv'):
    """Load the transaction dataset"""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print(f"Unique account_ids: {df['account_id'].nunique()}")
    return df


def prepare_features(df):
    """Extract features from the dataset"""
    print("\n" + "="*60)
    print("FEATURE PREPARATION")
    print("="*60)

    # Text columns
    text_columns = ['bank_description', 'merchant_name', 'note']
    print(f"\nCombining text columns: {text_columns}")
    df['combined_text'] = df[text_columns].fillna('').agg(' '.join, axis=1)
    print(f"  - Sample combined text: {df['combined_text'].iloc[0][:100]}...")

    # TF-IDF vectorization
    print(f"\nApplying TF-IDF vectorization (max_features=100)...")
    tfidf = TfidfVectorizer(max_features=100)
    text_features = tfidf.fit_transform(df['combined_text']).toarray()
    print(f"  - Text features shape: {text_features.shape}")
    print(f"  - Vocabulary size: {len(tfidf.vocabulary_)}")
    print(f"  - Sample feature names: {list(tfidf.get_feature_names_out())[:5]}")

    # Numeric features
    numeric_columns = ['credit_amount', 'debit_amount', 'balance_before_transaction',
                       'exchange_rate', 'exchange_rate_book']
    print(f"\nExtracting numeric features: {numeric_columns}")
    numeric_features = df[numeric_columns].fillna(0).values
    print(f"  - Numeric features shape: {numeric_features.shape}")
    print(f"  - Missing values filled with 0")

    # Show basic stats
    print(f"\nNumeric feature statistics:")
    for i, col in enumerate(numeric_columns):
        col_data = numeric_features[:, i]
        print(f"  - {col}: min={col_data.min():.2f}, max={col_data.max():.2f}, mean={col_data.mean():.2f}")

    # Combine features
    print(f"\nCombining text and numeric features...")
    X = np.hstack([text_features, numeric_features])

    # Encode account_id as categorical labels
    print(f"\nEncoding account_id as categorical labels...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['account_id'].values)

    print(f"\nFinal feature matrix shape: {X.shape}")
    print(f"  - Text features: {text_features.shape[1]} columns")
    print(f"  - Numeric features: {numeric_features.shape[1]} columns")
    print(f"  - Total features: {X.shape[1]} columns")
    print(f"\nTarget variable shape: {y.shape}")
    print(f"  - Number of classes: {len(np.unique(y))}")
    print(f"  - Label range: {y.min()} to {y.max()}")
    print(f"  - Original account_id range: {df['account_id'].min()} to {df['account_id'].max()}")
    print("="*60 + "\n")

    return X, y, len(label_encoder.classes_)


def get_data():
    """Main function to load and prepare data"""
    df = load_data()
    X, y, num_classes = prepare_features(df)
    return X, y, num_classes
