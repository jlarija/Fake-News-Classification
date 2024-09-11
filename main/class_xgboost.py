import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
import sys
import pandas as pd
import os 
import numpy as np 

class XGBoostWord2Vec(object):
    "class to train word2vec vectorization and xgboost ML approach"
    def __init__(self, path=None, sep=None):
        """Options to choose the separator and the path (both must be given as input in the terminal)
        """
        sys.stdout.write('Reading input file...\n')
        sys.stdout.flush()

        # Read the training dataframe
        self.df = pd.read_pickle(path)
        self.df_training = self.df[['vector', 'subject', 'Real']]

    def encode_features(self):
            sys.stdout.write('Encoding Labels for XGBoost...\n'); sys.stdout.flush()
            label_encoder = LabelEncoder()
            self.df_training['subject_encoded'] = label_encoder.fit_transform(self.df_training['subject'])

            # drop non encoded features
            self.df_training = self.df_training.drop(columns=['subject'])

            # Preprocess and encode features
            X_vectors = np.array(self.df_training['vector'].tolist())
            df_features = self.df_training.drop(columns=['vector', 'Real'])

            # Convert features to numpy array
            X_features = df_features.values

            # Combine word2vec vectors with features
            X_combined = np.hstack([X_vectors, X_features])
            y = self.df_training['Real'].values

            # Split the data into training and testing sets
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_combined, 
                y, 
                test_size=0.2, 
                random_state=42, 
                stratify=y
            )
            return
        
    def run_xgboost(self):
        sys.stdout.write('Training XGBoost...\n'); sys.stdout.flush()
        # initialize XGBoost classifier with class weights
        self.model_xgb = xgb.XGBClassifier(scale_pos_weight=2, random_state=42)
        self.model_xgb.fit(self.X_train, self.y_train)

        # make predictions on the test set
        self.y_pred = self.model_xgb.predict(self.X_test)

        sys.stdout.write('Model Trained. Parameters\n'); sys.stdout.flush()
        # print model evaluations
        accuracy = accuracy_score(self.y_test, self.y_pred)
        f1 = f1_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred)
        recall = recall_score(self.y_test, self.y_pred)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        return 
    
    def save_model(self, path_out):
        """Save the trained model to the specified path."""
        sys.stdout.write(f'Saving model to {path_out}...\n')
        sys.stdout.flush()
        self.model_xgb.save_model(path_out)
        sys.stdout.write('Model saved.\n')
        sys.stdout.flush()