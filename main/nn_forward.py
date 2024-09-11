'''Approach with a forward neural network with no hidden layers'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sys
import csv
class DataFrameLoader():
    """Class to load data and prepare it for training and evaluation."""
    def __init__(self, path):
        sys.stdout.write('Reading input file...\n')
        sys.stdout.flush()

        # Read the training dataframe
        self.df = pd.read_pickle(path)
        self.df_training = self.df[['vector', 'subject', 'Real']]
                
        # Convert features to numpy arrays
        X = np.array(self.df_training['vector'].tolist())  # Features - array of vectors
        y = np.array(self.df_training['Real'].tolist())    # Labels - array of 1s or 0s

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create Dataset instances
        self.train_dataset = VectorizedTextDataset(X_train, y_train)
        self.test_dataset = VectorizedTextDataset(X_test, y_test)

        # Create DataLoader instances
        self.train_loader = DataLoader(self.train_dataset, batch_size=2, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=2, shuffle=False)
    
class VectorizedTextDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label

class TextClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TextClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)  # Linear layer
        self.softmax = nn.Softmax(dim=1)  # Softmax activation for multi-class classification
    
    def forward(self, x):
        x = self.fc(x)  # Pass input through the linear layer
        return x  # Softmax is applied during evaluation

def train_model(model, train_loader, criterion, optimizer, epochs=5):
    sys.stdout.write('Starting Model Training...\n')
    with open('../output/loss_nn.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Epoch', 'Loss'])  # Write header
        for epoch in range(epochs):
            model.train()
            for features, labels in train_loader:
                optimizer.zero_grad()  # Clear previous gradients
                outputs = model(features)  # Forward pass
                loss = criterion(outputs, labels)  # Calculate loss
                loss.backward()  # Backward pass
                optimizer.step()  # Update weights
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')
            csv_writer.writerow([epoch + 1, loss.item()])

def evaluate_model(model, loader, path_out):
    '''Threshold is now kept at 0.5 but with the saved model it can be changed any time'''
    sys.stdout.write('Model Evaluation...\n')
    model.eval() # model is in evaluate mode
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in loader: # the loader is the dataset
            outputs = model(features) # get the outputs
            predicted = (outputs.squeeze() > 0.5).long()  # we only want to predict more than 0.8 - funny how you lose accuracy
            correct += (predicted == labels).sum().item() 
            total += labels.size(0)
    accuracy = correct / total
    # Save the model's state dictionary
    sys.stdout.write('Accuracy  ')
    print(accuracy)
    torch.save(model.state_dict(), path_out)
    sys.stdout.write('Model Saved in path_out\n')
    return 

