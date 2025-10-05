# according to 
# "Luz, T. S. F., Braga, R. A. S., & Ribeiro, E. R. (2024). Assessment of Ensemble-Based Machine Learning Algorithms for Exoplanet Identification. Electronics, 13(19), 3950. https://doi.org/10.3390/electronics13193950"
# we will use RandomForestClassifier for exoplanet identification with only 2 categories as target: CONFIRMED(1) and CANDIDATE(0).
# False Positives will be ingnored.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import joblib

class RandomForestModel:
    model_pipeline = None
    trained = False
    def __init__(self):
        # imputer with mean strategy
        imputer = SimpleImputer(strategy='mean')
        # scaler for feature scaling
        scaler = StandardScaler()
        # random forest classifier
        randomForest = RandomForestClassifier(random_state=42)
        # pipeline
        self.model_pipeline = Pipeline(steps=[
            ('imputer', imputer),
            #('scaler', scaler),
            ('classifier', randomForest)
        ])
        self.trained = False

    def load_data_source(self, data_source_path):
        self.data_source = pd.read_csv(data_source_path)

    def train(self, target_column, not_features=[], train_size=0.7, random_state=42):

        if self.data_source is None:
            raise ValueError("Data source not loaded. Please load data source before training.")
        
        X = self.data_source.drop(columns=[target_column] + not_features)
        y = self.data_source[target_column]
        print(  f"Features shape after preprocessing: {X.shape}")
        print(  f"Target shape after preprocessing: {y.shape}")
        print("Targets after preprocessing: ", y.value_counts())
        # split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=random_state)
        
        # fit the model
        self.model_pipeline.fit(X_train, y_train)
        
        # evaluate the model
        y_pred = self.model_pipeline.predict(X_test)
        print("Classification report:")
        print(classification_report(y_test, y_pred))
        print("Confusion matrix:")
        conf_mtrx = confusion_matrix(y_test, y_pred)
        print(conf_mtrx)
        precision = (conf_mtrx[0][0] + conf_mtrx[1][1]) / np.sum(conf_mtrx)
        print(f"Precision: {precision}")
        
        # cross-validation
        cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
        scores = cross_val_score(self.model_pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
        print(f"Cross-validated accuracy: {np.mean(scores)}")
        print(f"Standard deviation of accuracy: {np.std(scores)}")
        self.trained = True
        return {"confusion_matrix": conf_mtrx.tolist(), "precision": precision, "cv_accuracy": np.mean(scores), "cv_method":"RepeatedKFold(n_splits=5, n_repeats=5)", "cv_std": np.std(scores)}

    def predict(self, X):
        if not self.trained:
            raise ValueError("Model is not trained yet.")
        return self.model_pipeline.predict(X)
    
    def random_grid_search(self, target_column, not_features=[]):
        param_dist = {
            'n_estimators': randint(100, 500),
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': randint(2, 10),
            'min_samples_leaf': randint(1, 5),
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False],
            'class_weight': [None, 'balanced', 'balanced_subsample']
        }
        rf = RandomForestClassifier(random_state=42)
        random_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=50,
            scoring='f1_weighted',
            cv=5,
            verbose=2,
            random_state=42,
            n_jobs=-1
        )
        X = self.data_source.drop(columns=[target_column] + not_features)
        X = StandardScaler().fit_transform(X)
        y = self.data_source[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
        random_search.fit(X_train, y_train)
        print("Best Parameters:", random_search.best_params_)
        print("Best Score:", random_search.best_score_)
        self.model_pipeline = random_search.best_estimator_
        joblib.dump(self.model_pipeline, "./random_forest_model.joblib")
        self.trained = True
        return random_search.best_params_
    
    def export_model(self, model_path):
        if not self.trained:
            raise ValueError("Model is not trained yet.")
        joblib.dump(self.model_pipeline, model_path)
        print(f"Model saved to {model_path}")

if __name__ == "__main__":
    rf_model = RandomForestModel()
    rf_model.load_data_source("../raw-data/merged_data.csv")
    metrics = rf_model.train(target_column='disposition',not_features=[])
    print("Training metrics: ", metrics)
    # hyperparameter tuning doesn't seem to improve the model in this case
    #best_params = rf_model.random_grid_search(target_column='disposition', not_features=[])
    #print("Best hyperparameters from random grid search: ", best_params)
    rf_model.export_model("../models/random_forest_model.joblib")

    # export json
    import json
    with open("../models/random_forest_model_metrics.json", "w") as f:
        json.dump(metrics, f)