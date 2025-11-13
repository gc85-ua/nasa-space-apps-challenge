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
from tabulate import tabulate

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
        
        # Get classification metrics
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        conf_mtrx = confusion_matrix(y_test, y_pred)
        
        # cross-validation
        cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
        scores = cross_val_score(self.model_pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
        cv_mean = np.mean(scores)
        cv_std = np.std(scores)
        
        # Print results as formatted tables
        print("\n" + "="*70)
        print("MODEL TRAINING RESULTS")
        print("="*70)
        
        # Dataset information table
        print("\n📊 DATASET INFORMATION:")
        dataset_info = [
            ["Training samples", X_train.shape[0]],
            ["Testing samples", X_test.shape[0]],
            ["Number of features", X_train.shape[1]],
            ["Total samples", X.shape[0]]
        ]
        print(tabulate(dataset_info, headers=["Metric", "Value"], tablefmt="grid"))
        
        # Classification metrics table
        print("\n📈 CLASSIFICATION METRICS:")
        metrics_table = [
            ["Accuracy", f"{accuracy:.4f}"],
            ["Precision", f"{precision:.4f}"],
            ["Recall", f"{recall:.4f}"],
            ["F1-Score", f"{f1:.4f}"]
        ]
        print(tabulate(metrics_table, headers=["Metric", "Score"], tablefmt="grid"))
        
        # Confusion matrix
        print("\n🎯 CONFUSION MATRIX:")
        cm_data = [
            ["True Negatives", conf_mtrx[0][0], "False Positives", conf_mtrx[0][1]],
            ["False Negatives", conf_mtrx[1][0], "True Positives", conf_mtrx[1][1]]
        ]
        print(tabulate(cm_data, headers=["", "Predicted Negative", "", "Predicted Positive"], tablefmt="grid"))
        
        # Cross-validation results table
        print("\n🔄 CROSS-VALIDATION RESULTS (RepeatedKFold: 5 splits, 5 repeats):")
        cv_table = [
            ["Mean Accuracy", f"{cv_mean:.4f}"],
            ["Standard Deviation", f"{cv_std:.4f}"]
        ]
        print(tabulate(cv_table, headers=["Metric", "Value"], tablefmt="grid"))
        
        print("\n" + "="*70 + "\n")
        
        self.trained = True
        return {
            "confusion_matrix": conf_mtrx.tolist(), 
            "precision": precision, 
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "cv_accuracy": cv_mean, 
            "cv_method": "RepeatedKFold(n_splits=5, n_repeats=5)", 
            "cv_std": cv_std
        }

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
        print("\n" + "="*70)
        print("HYPERPARAMETER TUNING RESULTS (RandomizedSearchCV)")
        print("="*70)
        
        print("\n🔧 BEST PARAMETERS FOUND:")
        best_params_list = [[key, value] for key, value in random_search.best_params_.items()]
        print(tabulate(best_params_list, headers=["Parameter", "Value"], tablefmt="grid"))
        
        print("\n✅ BEST SCORE:", f"{random_search.best_score_:.4f}")
        print("="*70 + "\n")
        
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
    
    # Print summary table of all metrics
    print("\n" + "="*70)
    print("FINAL METRICS SUMMARY")
    print("="*70)
    summary_table = [
        ["Accuracy", f"{metrics['accuracy']:.4f}"],
        ["Precision", f"{metrics['precision']:.4f}"],
        ["Recall", f"{metrics['recall']:.4f}"],
        ["F1-Score", f"{metrics['f1_score']:.4f}"],
        ["Cross-Validation Accuracy", f"{metrics['cv_accuracy']:.4f}"],
        ["CV Std Deviation", f"{metrics['cv_std']:.4f}"]
    ]
    print(tabulate(summary_table, headers=["Metric", "Value"], tablefmt="fancy_grid"))
    print("="*70 + "\n")
    
    # hyperparameter tuning doesn't seem to improve the model in this case
    #best_params = rf_model.random_grid_search(target_column='disposition', not_features=[])
    #print("Best hyperparameters from random grid search: ", best_params)
    rf_model.export_model("../models/random_forest_model.joblib")

    # export json
    import json
    with open("../models/random_forest_model_metrics.json", "w") as f:
        json.dump(metrics, f)