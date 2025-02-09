import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

class CNNRNNAutoencoder(nn.Module):
    def __init__(self, num_classes, seq_len=161, input_channels=6):
        super().__init__()
        
        self.seq_len = seq_len
        self.input_channels = input_channels
        
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.3)
        )
        
        self.rnn = nn.LSTM(input_size=input_channels, hidden_size=64, num_layers=3, batch_first=True, dropout=0.3)
        
        cnn_output_size = 64 * (seq_len // 4)
        
        self.encoder = nn.Sequential(
            nn.Linear(cnn_output_size + 64, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, input_channels * seq_len)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        x_cnn = self.cnn(x)
        cnn_feature = x_cnn.view(batch_size, -1)
        
        x_rnn = x.permute(0, 2, 1)
        _, (rnn_feature, _) = self.rnn(x_rnn)
        rnn_feature = rnn_feature[-1]
        
        combined_feature = torch.cat((cnn_feature, rnn_feature), dim=1)
        
        encoded = self.encoder(combined_feature)
        decoded = self.decoder(encoded)
        
        decoded = decoded.view(batch_size, self.input_channels, self.seq_len)
        
        return encoded, decoded

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            encoded, decoded = model(inputs)
            loss = criterion(decoded, inputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(device)
                encoded, decoded = model(inputs)
                loss = criterion(decoded, inputs)
                val_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')

def extract_features(model, data_loader, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            _, decoded= model(inputs)
            features.append(decoded.cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(features), np.concatenate(labels)

dataset_data = np.load('eeg_dataset_data.npy')
dataset_labels = np.load('eeg_dataset_labels.npy') - 1

X_train, X_test, y_train, y_test = train_test_split(dataset_data, dataset_labels, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)

X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.LongTensor(y_test)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_classes = len(np.unique(dataset_labels))
input_channels = X_train.shape[1]
seq_len = X_train.shape[2]
model = CNNRNNAutoencoder(num_classes, seq_len, input_channels)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 50
train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device)

train_features, train_labels = extract_features(model, train_loader, device)
test_features, test_labels = extract_features(model, test_loader, device)

xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(train_features, train_labels)

predictions = xgb_model.predict(test_features)

accuracy = accuracy_score(test_labels, predictions)
print(f"Final Test Accuracy: {accuracy:.4f}")