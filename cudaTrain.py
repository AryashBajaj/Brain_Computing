import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import preprocessing
import scipy.io as sc
import numpy as np
import random
import xgboost as xgb
import time

# Check for CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# this function is used to transfer one column label to one hot label
def one_hot(y_):
    y_ = y_.reshape(len(y_))
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]

# Data loading
feature = sc.loadmat("S1_nolabel6.mat")
all_data = feature['S1_nolabel6']

np.random.shuffle(all_data)

final = 2800 * 10
all_data = all_data[0:final]
features = all_data[:, 0:64]
labels = all_data[:, 64:65]

# Z-score normalization
features = preprocessing.scale(features)
labels_onehot = one_hot(labels)

n_classes = 6
n_fea = features.shape[-1]
middle_number = int(final * 3 / 4)
print(n_fea)

# Split the data
features_training = features[:middle_number]
features_testing = features[middle_number:]
labels_training = labels_onehot[:middle_number]
labels_testing = labels_onehot[middle_number:]

# Convert to tensors and transfer to device
features_training = torch.tensor(features_training, dtype=torch.float32).to(device)
features_testing = torch.tensor(features_testing, dtype=torch.float32).to(device)
labels_training = torch.tensor(labels_training, dtype=torch.float32).to(device)
labels_testing = torch.tensor(labels_testing, dtype=torch.float32).to(device)

# Define CNN model
class CNN(nn.Module):
    def __init__(self, n_fea, n_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=(1, 1), padding='same')
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.fc1 = nn.Linear(n_fea // 2 * 20, 120)
        self.fc2 = nn.Linear(120, n_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.sigmoid(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Instantiate CNN model, move to device, loss function, and optimizer
cnn_model = CNN(n_fea, n_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.04)

# Training CNN
n_epochs = 1500
batch_size = final - middle_number

train_dataset = torch.utils.data.TensorDataset(features_training, labels_training)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

def compute_accuracy(model, features, labels):
    with torch.no_grad():
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)
        _, true_labels = torch.max(labels, 1)
        correct = (predicted == true_labels).sum().item()
        accuracy = correct / labels.size(0)
    return accuracy

for epoch in range(n_epochs):
    for i, (batch_features, batch_labels) in enumerate(train_loader):
        optimizer.zero_grad()
        print(batch_features.shape)
        print(batch_features.reshape(batch_features.shape[0], 1, 1, batch_features.shape[-1]).shape)
        outputs = cnn_model(batch_features.reshape(batch_features.shape[0], 1, 1, batch_features.shape[-1]))  # Ensure correct dimension for conv layer
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        print("cnn")

    if epoch % 5 == 0:
        accuracy = compute_accuracy(cnn_model, features_testing.reshape(features_testing.shape[0], 1, 1, features_testing.shape[-1]), labels_testing)
        print(f"Epoch {epoch}, Loss: {loss.item()}, Accuracy: {accuracy}")

# Save CNN model
torch.save(cnn_model.state_dict(), 'cnn_model.pth')

# Extract features from CNN
cnn_model.eval()
with torch.no_grad():
    cnn_features = cnn_model(features_training.unsqueeze(1))
    cnn_features_test = cnn_model(features_testing.unsqueeze(1))

# RNN
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), nodes).to(x.device)
        c0 = torch.zeros(2, x.size(0), nodes).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take the output from the last time step
        return out

nodes = 264
rnn_model = RNN(n_fea, nodes, n_classes).to(device)
optimizer_rnn = optim.Adam(rnn_model.parameters(), lr=0.005)

# Training RNN
rnn_model.train()
for epoch in range(n_epochs):
    for i, (batch_features, batch_labels) in enumerate(train_loader):
        optimizer_rnn.zero_grad()
        rnn_output = rnn_model(batch_features.unsqueeze(1))
        loss = criterion(rnn_output, batch_labels)
        loss.backward()
        optimizer_rnn.step()

    if epoch % 5 == 0:
        rnn_accuracy = compute_accuracy(rnn_model, features_testing.unsqueeze(1), labels_testing)
        print(f"Epoch {epoch}, RNN Accuracy: {rnn_accuracy}, Loss: {loss.item()}")

# Save RNN model
torch.save(rnn_model.state_dict(), 'rnn_model.pth')

# Extract features from RNN
rnn_model.eval()
with torch.no_grad():
    rnn_features = rnn_model(features_training.unsqueeze(1))
    rnn_features_test = rnn_model(features_testing.unsqueeze(1))

# Combine CNN and RNN features
combined_features = torch.cat((cnn_features, rnn_features), dim=1)
combined_features_test = torch.cat((cnn_features_test, rnn_features_test), dim=1)

# Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.Sigmoid(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size2, hidden_size1),
            nn.Sigmoid(),
            nn.Linear(hidden_size1, input_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

n_input_ae = combined_features.size(1)
ae_model = Autoencoder(n_input_ae, 800, 100).to(device)
optimizer_ae = optim.RMSprop(ae_model.parameters(), lr=0.2)

# Train Autoencoder
for epoch in range(n_epochs):
    for i, (batch_features, _) in enumerate(train_loader):
        optimizer_ae.zero_grad()
        encoded, decoded = ae_model(batch_features)
        loss = nn.MSELoss()(decoded, batch_features)
        loss.backward()
        optimizer_ae.step()

    if epoch % 5 == 0:
        print(f"Epoch {epoch}, AE Loss: {loss.item()}")

# Save Autoencoder model
torch.save(ae_model.state_dict(), 'autoencoder_model.pth')

# XGBoost
xg_train = xgb.DMatrix(ae_model.encoder(combined_features).cpu().detach().numpy(), label=np.argmax(labels_training.cpu(), 1))
xg_test = xgb.DMatrix(ae_model.encoder(combined_features_test).cpu().detach().numpy(), label=np.argmax(labels_testing.cpu(), 1))

param = {
    'objective': 'multi:softprob',
    'eta': 0.5,
    'max_depth': 6,
    'silent': 1,
    'nthread': 4,
    'subsample': 0.9,
    'num_class': n_classes
}

watchlist = [(xg_train, 'train'), (xg_test, 'test')]
num_round = 500
bst = xgb.train(param, xg_train, num_round, watchlist)

# Save XGBoost model
bst.save_model('xgboost_model.json')
