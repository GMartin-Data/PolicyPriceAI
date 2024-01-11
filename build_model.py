import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import uniform
import wget

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler


# Utility Function for Data Preprocessing
def split_bmi_in_three(x: float) -> str:
    if x < 25:
        return "underweight_normal"
    if x < 30:
        return "overweight"
    return "obesity"


# Creating Folder's Structure in Case of First Run
dirs = ["csvs", "graphs"]
for dir in dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)

# Loading Data
url = "https://simplonline-v3-prod.s3.eu-west-3.amazonaws.com/media/file/csv/4072eb5e-e963-4a17-a794-3ea028d0a9c4.csv"
output_path = "csvs/dataset.csv"
wget.download(url, output_path)

# Cleaning Data
df = pd.read_csv(output_path).drop_duplicates()

# Dump Clean Data
df.to_csv("csvs/cleaned_dataset.csv", index=False)

# Separating Target and Features
y = df.pop("charges")
X = df

# Modifying Target's Shape
y = np.log(y + 1)

# Hold-Out
X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=True, train_size=0.85, random_state=42, stratify=X["smoker"]
)

# Build Pipeline
ohe_nom = OneHotEncoder(drop="first", handle_unknown="ignore")
ohe_bin = OneHotEncoder(drop="if_binary", handle_unknown="ignore")
poly = PolynomialFeatures(degree=2)
std = StandardScaler()

en = ElasticNet(random_state=42, max_iter=10_000, tol=1e-3)

encoder = ColumnTransformer(
    transformers=[
        ("bin", ohe_bin, ["sex", "smoker"]),
        ("nom", ohe_nom, ["bmi", "region"]),
    ],
    remainder="passthrough",
)

model = make_pipeline(encoder, poly, std, en)

# Binning bmi outside Pipeline
X_bmi_nom = X.copy()
X_bmi_nom.bmi = X_bmi_nom.bmi.apply(split_bmi_in_three)

# Stratified (smoker) Hold-Out
X_bmi_nom_train, X_bmi_nom_test, y_train, y_test = train_test_split(
    X_bmi_nom, y, shuffle=True, train_size=0.85, random_state=42, stratify=X["smoker"]
)

# Train Model
params = {"elasticnet__alpha": uniform(0, 2), "elasticnet_l1_ratio": uniform(0, 1)}

random_search = RandomizedSearchCV(
    model, param_distributions=params, n_iter=2_000, cv=5, n_jobs=-1
)

random_search.fit(X_bmi_nom_train, y_train)

# Score
best_model = random_search.best_estimator_
best_model.fit(X_bmi_nom_train, y_train)
best_model.score(X_bmi_nom_test, y_test)

# Save the model
joblib.dump(best_model, "model.joblib")
