from model import CNN
from data_handler import CustomDataset
from torch.utils.data import random_split, DataLoader
import torch
import torch.nn as nn

dataset = CustomDataset('./images')

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=4)

print("Classes:", dataset.classes)
print(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")



def train(model, train_loader, criterion, optimizer, device, epochs=10):
    model.to(device)
    model.train()
    i = 0
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()           # clear gradients
            outputs = model(inputs)         # forward pass
            loss = criterion(outputs, labels)  # compute loss
            
            loss.backward()                 # backward pass
            optimizer.step()                # update weights
            
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print(epoch + i)
            i = i + 1
        
        scheduler.step()

        epoch_loss = running_loss / total
        epoch_acc = correct / total * 100
        
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")


model = CNN(num_classes=36)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# CALL FUNCTION FOR TRAINING

if __name__ == "__main__":
    train(model, train_loader, criterion, optimizer, device, epochs=30)