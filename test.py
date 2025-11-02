import torch
import torch.nn as nn
from model import CNN
from data_handler import test_loader

model = CNN(num_classes=8)
criterion = nn.CrossEntropyLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.load_state_dict(torch.load('model.pth', map_location=device))

def test(model, test_loader, criterion, device):
    model.to(device)
    model.eval() 

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad(): 
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print(total, correct)

    avg_loss = running_loss / total
    accuracy = correct / total * 100
    print(f"Test Loss: {avg_loss:.4f} - Test Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

# CALL FUNCTION FOR TESTING

if __name__ == "__main__":
    test(model, test_loader, criterion, device)
