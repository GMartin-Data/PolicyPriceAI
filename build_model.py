import os
from typing import List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import uniform
import wget

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler


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


# Creating Folder's Structure in Case of First Run
dirs = ["csvs", "graphs"]
for dir in dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)

# Loading Data
data_path = "csvs/dataset.csv"
if not os.path(data_path):
    url = "https://simplonline-v3-prod.s3.eu-west-3.amazonaws.com/media/file/csv/4072eb5e-e963-4a17-a794-3ea028d0a9c4.csv"
    wget.download(url, data_path)

# Cleaning Data
df = pd.read_csv(data_path).drop_duplicates()

# Dump Clean Data
clean_path = "csvs/cleaned_dataset.csv"
if not os.path(clean_path):
    df.to_csv(clean_path, index=False)

# Separating Target and Features
y = df.pop("charges")
X = df

# Modifying Target's Shape
y = np.log(y + 1)

# Hold-Out, with stratified k-fold on `smoker`
X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=True, train_size=0.85, random_state=42, stratify=X["smoker"]
)

# Building Pipeline
# Instanciate custom transformer for `bmi`
bmi_edges = [0.0, 25.0, 30.0, np.inf]
bmi_cats = ["underweight_normal", "overweight", "obesity"]
bmi_categorizer = ThresholdBinningTransformer(column="bmi", bins=bmi_edges, labels=bmi_cats)   

ohe_nom = OneHotEncoder(drop="first", handle_unknown="ignore")
ohe_bin = OneHotEncoder(drop="if_binary", handle_unknown="ignore")

pipe_bmi = make_pipeline(bmi_categorizer, ohe_nom)

# Stages of the Pipeline
encoder = ColumnTransformer(
    transformers=[
        ("bmi", pipe_bmi, ["bmi"]),
        ("bin", ohe_bin, ["sex", "smoker"]),
        ("nom", ohe_nom, ["region"]),
    ],
    remainder="passthrough",
)
poly = PolynomialFeatures(degree=2)
std = StandardScaler()
en = ElasticNet(random_state=42, max_iter=10_000, tol=1e-3)

model = make_pipeline(encoder, poly, std, en)

# Train Model
params = {"elasticnet__alpha": uniform(0, 2), "elasticnet__l1_ratio": uniform(0, 1)}

random_search = RandomizedSearchCV(
    model, param_distributions=params, n_iter=2_000, cv=5, n_jobs=-1
)

random_search.fit(X_train, y_train)

# Score
best_model = random_search.best_estimator_
best_model.fit(X_train, y_train)
score = best_model.score(X_test, y_test)
print(f"Score on test set: {score}")

# Save the model
joblib.dump(best_model, "model.joblib")
