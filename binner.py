from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# Utility Class for Data Preprocessing
class ThresholdBinningTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column: str, bins: List[float], labels=List[str]):
        self.column = column
        self.bins = bins
        self.labels = labels

    def fit(self, X, y=None):
        # No fitting necessary for this transformer
        return self

    def transform(self, X):
        if self.column in X.columns:
            X_binned = pd.cut(X[self.column],
                              bins=self.bins, labels=self.labels,
                              right=False)  # left edge inclusive, right edge exclusive
            X_transformed = X.copy()
            X_transformed[self.column] = X_binned
            return X_transformed
        else:
            raise ValueError(f"Column {self.column} not in input") 