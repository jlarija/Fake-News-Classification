'''Approach with a forward neural network with no hidden layers'''
import torch
import torch.nn as nn
import torch.optim as optim

class TextClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TextClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)  # Linear layer
        self.softmax = nn.Softmax(dim=1)  # Softmax activation for multi-class classification
    
    def forward(self, x):
        x = self.fc(x)  # Pass input through the linear layer
        return x  # Softmax is applied during evaluation

def train_model(model, train_loader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        model.train()
        for features, labels in train_loader:
            optimizer.zero_grad()  # Clear previous gradients
            outputs = model(features)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in loader:
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)  # Get the predicted class
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    # Save the model's state dictionary
    torch.save(model.state_dict(), 'model.pth')
    return accuracy