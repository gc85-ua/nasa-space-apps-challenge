# according to 
# "Luz, T. S. F., Braga, R. A. S., & Ribeiro, E. R. (2024). Assessment of Ensemble-Based Machine Learning Algorithms for Exoplanet Identification. Electronics, 13(19), 3950. https://doi.org/10.3390/electronics13193950"
# we will use RandomForestClassifier for exoplanet identification with only 2 categories as target: CONFIRMED(1) and CANDIDATE(0).
# False Positives will be ingnored.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

plt.interactive(True)

# imputer with mean strategy
imputer = SimpleImputer(strategy='mean')
# scaler for feature scaling
scaler = StandardScaler()
# random forest classifier
randomForest = RandomForestClassifier(random_state=42)
# pipeline
pipeline = Pipeline(steps=[
    ('imputer', imputer),
    ('scaler', scaler),
    ('classifier', randomForest)
])

target = "koi_disposition"
not_features = ["kepid","kepoi_name", "kepler_name", "koi_pdisposition", "koi_score", "koi_tce_plnt_num", "koi_tce_delivname"]
empty_cols = ["koi_teq_err1", "koi_teq_err2"]

# load source data
source_data_df = pd.read_csv("../raw-data/cumulative.csv")

fp_flag_cols = [col for col in source_data_df.columns if 'fpflag' in col]
print("Initial targets: ", source_data_df[target].value_counts())
# prepocessing source data, dropping rows with koi_disposition = FALSE POSITIVE
source_data_df = source_data_df[source_data_df[target] != "FALSE POSITIVE"]
# preprocessing source data, converting target column to binary values
source_data_df[target] = source_data_df[target].apply(lambda x: 1 if x == "CONFIRMED" else 0)

# obtaining features and target
X = source_data_df.drop(columns=[target] + not_features + empty_cols + fp_flag_cols)
y = source_data_df[target]

print(  f"Features shape after preprocessing: {X.shape}")
print(  f"Target shape after preprocessing: {y.shape}")
print("Targets after preprocessing: ", y.value_counts())

# splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

# eval model without hyperparameter tuning
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("Classification report:")
print(classification_report(y_test, y_pred))
print("Confusion matrix:")
conf_mtrx = confusion_matrix(y_test, y_pred)
print(conf_mtrx)
precission = (conf_mtrx[0][0] + conf_mtrx[1][1]) / np.sum(conf_mtrx)
print(f"Precission: {precission}")

# cross-validation
cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=42)
scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print(f"Cross-validated accuracy: {np.mean(scores)}")
print(f"Standard deviation of accuracy: {np.std(scores)}")