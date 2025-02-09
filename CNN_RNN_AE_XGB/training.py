import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from torchvision.transforms import ToTensor

class CNNRNNAutoencoder(nn.Module):
    def _init_(self, num_classes, seq_len=129, input_channels=8):  
        super(CNNRNNAutoencoder, self)._init_()
        
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
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(64 * (seq_len // 4), 120)
        )
        #fc1 and fc2 layers are missing.
        self.rnn = nn.LSTM(input_size=input_channels, hidden_size=64, num_layers=3, batch_first=True, dropout=0.3)
        
        self.encoder = nn.Sequential(
            nn.Linear(120 + 64, 256),
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
            nn.Linear(256, 184)
        )

        self.imageMaker = nn.Linear(184, 129 * 8)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        x_cnn = torch.sigmoid(self.cnn(x))
        cnn_feature = x_cnn
        
        x_rnn = x.permute(0, 2, 1)
        rnn_feature, _ = self.rnn(x_rnn)
        rnn_feature = rnn_feature[:,-1,:]
        
        combined_feature = torch.cat((cnn_feature, rnn_feature), dim=1)
        
        encoded = self.encoder(combined_feature)
        decoded = self.decoder(encoded)
        decoded_signal = self.imageMaker(decoded).view(batch_size, input_channels, 161)
        return decoded, decoded_signal

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, xgb_model, train_labels):
    model.to(device)
    all_features = []
    for epoch in range(num_epochs):
        model.train()
        training_loss = 0.0
        for inputs, _ in train_loader:
            inputs = inputs.to(device)  # Move inputs to GPU
            optimizer.zero_grad()
            decoded, signal = model(inputs)
            if (epoch == 0) :
                all_features.append(decoded.detach().cpu().numpy())
            loss = criterion(signal, inputs)
            training_loss += loss.item()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(device)  # Move inputs to GPU
                _, signal = model(inputs)
                loss = criterion(signal, inputs)
                val_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {training_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')
    
    all_features = np.concatenate(all_features)
    print(all_features.shape)
    
    # Save the trained model
    torch.save(model.state_dict(), "cnn_rnn_autoencoder3.pth")
    print("Model saved as 'cnn_rnn_autoencoder3.pth'")
    train_labels = torch.tensor(train_labels)
    #train_labels = torch.nn.functional.one_hot(train_labels)
    xgb_model.fit(all_features, train_labels)



def extract_features(model, data_loader, device):
    model.eval()
    model.to(device)
    features = []
    labels = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)  # Move inputs to GPU
            _, decoded = model(inputs)
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
print(y_test_tensor.shape, X_test_tensor.shape)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_classes = len(np.unique(dataset_labels))
print(X_train.shape)
input_channels = X_train.shape[1]
seq_len = X_train.shape[2]
model = CNNRNNAutoencoder(num_classes, seq_len, input_channels)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 10
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
train_features, train_labels = extract_features(model, train_loader, device)
train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device, xgb_model, train_labels)

test_features, test_labels = extract_features(model, test_loader, device)
testSet = DataLoader(train_dataset, batch_size=32, shuffle=True)

all_decoded = []
model.to(device)
for inputs, labels in testSet :
    inputs=inputs.cuda()
    decoded, _ = model(inputs)

    all_decoded.append(decoded.detach().cpu().numpy())

all_decoded = np.concatenate(all_decoded)
    
predictions = xgb_model.predict(all_decoded)
#test_labels = torch.nn.functional.one_hot(torch.tensor(test_labels))
accuracy = accuracy_score(train_labels, predictions)
print(f"Final Test Accuracy: {accuracy:.4f}")