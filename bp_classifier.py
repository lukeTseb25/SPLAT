import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

class TanhLogNormalizer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that applies tanh(ln(x+1)) normalization to specified columns.
    """
    def __init__(self, columns_to_normalize):
        self.columns_to_normalize = columns_to_normalize
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.columns_to_normalize:
            X[:, col] = np.tanh(np.log(X[:, col] + 1))
        return X


def load_data(data_path, labels_path):
    """Load features and labels from CSV files."""
    X = pd.read_csv(data_path, header=None).values
    y = pd.read_csv(labels_path, header=None).values.ravel()
    return X, y


def create_pipeline():
    """Create preprocessing and classifier pipeline."""
    # Columns 16-31 need the special normalization
    columns_to_normalize = list(range(16, 32))
    
    pipeline = Pipeline([
        ('tanh_log_normalize', TanhLogNormalizer(columns_to_normalize)),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    return pipeline


def main():
    print("Loading data...")
    X, y = load_data(
        'data/processed/output_MI_EEG_20251005_171205_Session1LS.csv',
        'data/processed/labels_MI_EEG_20251005_171205_Session1LS.csv'
    )
    
    print(f"Data shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Create and train pipeline
    print("\nTraining classifier...")
    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y)))
    plt.xticks(tick_marks, tick_marks)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("\nConfusion matrix saved as 'confusion_matrix.png'")
    
    # Feature importance
    feature_importance = pipeline.named_steps['classifier'].feature_importances_
    print(f"\nTop 10 important features:")
    top_indices = np.argsort(feature_importance)[-10:][::-1]
    for idx in top_indices:
        print(f"  Feature {idx}: {feature_importance[idx]:.4f}")
    
    return pipeline, (X_test, y_test, y_pred)


if __name__ == "__main__":
    pipeline, results = main()
