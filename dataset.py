"""
Dataset loading and feature extraction for Forest Covertype classification
"""
import pandas as pd
import numpy as np
import gzip
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(filepath='data/covertype/covtype.csv'):
    """Load the Forest Covertype dataset, auto-decompressing if needed"""
    csv_path = Path(filepath)
    gz_path = Path(f"{filepath}.gz")
    
    # If CSV doesn't exist but .gz does, decompress it
    if not csv_path.exists() and gz_path.exists():
        print(f"Decompressing {gz_path}...")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(csv_path, 'wb') as f_out:
                f_out.write(f_in.read())
        print(f"✓ Decompressed to {csv_path}")
    
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")
    return df


def prepare_features(df):
    """Extract features from the dataset"""
    print("\n" + "="*60)
    print("FEATURE PREPARATION")
    print("="*60)

    # Separate features and target
    feature_columns = [col for col in df.columns if col != 'Cover_Type']
    X = df[feature_columns].values
    y = df['Cover_Type'].values
    
    # Note: Cover_Type is already 1-7, but we'll convert to 0-6 for PyTorch
    y = y - 1  # Convert from 1-7 to 0-6
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"  - Total features: {X.shape[1]} columns")
    print(f"  - Feature breakdown:")
    print(f"    * 10 quantitative variables (columns 0-9)")
    print(f"    * 4 wilderness areas (columns 10-13)")
    print(f"    * 40 soil types (columns 14-53)")
    
    print(f"\nTarget variable (Cover_Type):")
    print(f"  - Shape: {y.shape}")
    print(f"  - Number of classes: {len(np.unique(y))} (converted from 1-7 to 0-6)")
    print(f"  - Label range: {y.min()} to {y.max()}")
    
    # Scale features (standardization)
    print(f"\nScaling features (StandardScaler)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"  - Features scaled to mean=0, std=1")
    
    print("="*60 + "\n")

    return X_scaled, y, len(np.unique(y)), scaler


def get_data(test_size=0.2, val_size=0.1, random_state=42):
    """
    Main function to load and prepare data with train/val/test splits
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, num_classes, scaler
    """
    df = load_data()
    X, y, num_classes, scaler = prepare_features(df)
    
    # First split: separate test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: separate train and validation from train_val
    val_size_adjusted = val_size / (1 - test_size)  # Adjust for already split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size_adjusted, 
        random_state=random_state, stratify=y_train_val
    )
    
    print(f"Data splits:")
    print(f"  - Train: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
    print(f"  - Val:   {X_val.shape[0]:,} samples ({X_val.shape[0]/len(df)*100:.1f}%)")
    print(f"  - Test:  {X_test.shape[0]:,} samples ({X_test.shape[0]/len(df)*100:.1f}%)")
    print()
    
    return X_train, X_val, X_test, y_train, y_val, y_test, num_classes, scaler
