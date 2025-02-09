from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from sklearn.metrics import accuracy_score

class NeuralNetwork(nn.Module):
    def __init__(self, num_classes, seq_len=129):
        super().__init__()
        
        self.seq_len = seq_len
        
        self.conv1 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(4096, 512)
        self.fc2 = nn.Linear(512, 120)
        self.drop = nn.Dropout(0.5)
        
        self.rnn = nn.LSTM(input_size=128, hidden_size=64, num_layers=5, batch_first=True)
        
        self.encoder = nn.Linear(184, 120)
        self.decoder = nn.Linear(120, 184)
        
        self.fc3 = nn.Linear(184, 100)
        self.fc4 = nn.Linear(100, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        batch_size = x.size(0)
        x_cnn = self.pool1(self.drop(torch.relu(self.conv1(x))))
        x_cnn = self.pool2(self.drop(torch.relu(self.conv2(x_cnn))))
        cnn_feature = x_cnn.view(batch_size, -1)
        cnn_feature = self.fc1(cnn_feature)
        cnn_feature = self.fc2(cnn_feature)
        x_rnn = x_cnn.permute(0, 2, 1)
        rnn_out, _ = self.rnn(x_rnn)
        rnn_feature = rnn_out[:, -1, :] 
        combined_feature = torch.cat((rnn_feature, cnn_feature), dim=1)
        encoded = torch.relu(self.encoder(combined_feature))
        decoded = torch.relu(self.decoder(encoded))
        output = self.fc3(decoded)
        output = self.softmax(self.fc4(output))
        return output
    
model = NeuralNetwork(3)
loss_fn = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

dataset_data = np.load('eeg_dataset_data.npy')
dataset_labels = np.load('eeg_dataset_labels.npy') - 1

train_size = int(0.8 * len(dataset_data))
test_size = len(dataset_data) - train_size

train_data = dataset_data[:train_size]
train_labels = dataset_labels[:train_size]

test_data = dataset_data[train_size:]
test_labels = dataset_labels[train_size:]

scaler = StandardScaler()

train_data_scaled = scaler.fit_transform(train_data.reshape(train_data.shape[0], -1))
train_data_scaled = train_data_scaled.reshape(train_data.shape)

test_data_scaled = scaler.transform(test_data.reshape(test_data.shape[0], -1))
test_data_scaled = test_data_scaled.reshape(test_data.shape)

train_data_tensor = torch.tensor(train_data_scaled, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)

test_data_tensor = torch.tensor(test_data_scaled, dtype=torch.float32)
test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
test_dataset = TensorDataset(test_data_tensor, test_labels_tensor)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for batch_data, batch_labels in train_loader:
    print("Batch data shape:", batch_data.shape)
    print("Batch labels shape:", batch_labels.shape)
    break

epochs = 20

for epoch in range(epochs):
    model.train()
    running_loss = 0
    for batch_data, batch_labels in train_loader:
        optimiser.zero_grad()
        outputs = model(batch_data)
        loss = loss_fn(outputs, batch_labels)
        loss.backward()
        optimiser.step()
        running_loss += loss.item()

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for test_data, test_labels in test_loader:
            outputs = model(test_data)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(test_labels.numpy())

    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Epoch {epoch + 1}/{epochs} - Loss: {running_loss / len(train_loader):.4f} - Accuracy: {accuracy:.4f}")

torch.save(model.state_dict(), 'cnn_model.pth')
