from model import CNN
import torch
import torch.nn as nn
from data_handler import train_loader

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

            if i % 30 == 0:  # save every 30 batches
                torch.save(model.state_dict(), 'model.pth')
        
        scheduler.step()

        epoch_loss = running_loss / total
        epoch_acc = correct / total * 100
        
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")


model = CNN(num_classes=8)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# CALL FUNCTION FOR TRAINING

if __name__ == "__main__":
    train(model, train_loader, criterion, optimizer, device, epochs=30)
