from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
import seaborn as sns
import matplotlib.pyplot as plt

class NeuralNetwork(nn.Module):
    def __init__(self, num_classes, seq_len=129):
        super().__init__()
        
        self.seq_len = seq_len
        
        self.conv1 = nn.Conv1d(in_channels=8, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * (seq_len // 4), 128)
        
        self.rnn = nn.LSTM(input_size=64, hidden_size=32, num_layers=2, batch_first=True)
        
        self.fc2 = nn.Linear(128 + 32, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x_cnn = self.pool1(nn.functional.relu(self.conv1(x)))
        x_cnn = self.pool2(nn.functional.relu(self.conv2(x_cnn)))
        
        cnn_feature = x_cnn.view(batch_size, -1)
        cnn_feature = nn.functional.relu(self.fc1(cnn_feature))

        x_rnn = x_cnn.permute(0, 2, 1) 
        rnn_out, _ = self.rnn(x_rnn)
        rnn_feature = rnn_out[:, -1, :]
        
        combined_feature = torch.cat((cnn_feature, rnn_feature), dim=1)
        
        output = self.fc2(combined_feature)
        return output

def main() :
    dataset_data = np.load('Data_files\\eeg_dataset_data.npy')
    dataset_labels = np.load('Data_files\\eeg_dataset_labels.npy') - 1

    ros = RandomOverSampler(random_state=42)
    dataset_data_reshaped = dataset_data.reshape(dataset_data.shape[0], -1)

    data_resampled, labels_resampled = ros.fit_resample(dataset_data_reshaped, dataset_labels)

    train_size = int(0.75 * len(data_resampled))
    train_data = data_resampled[:train_size]
    train_labels = labels_resampled[:train_size]
    test_data = data_resampled[train_size:]
    test_labels = labels_resampled[train_size:]

    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)

    train_data_tensor = torch.tensor(train_data_scaled, dtype=torch.float32).view(-1, 8, 129) 
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)

    test_data_tensor = torch.tensor(test_data_scaled, dtype=torch.float32).view(-1, 8, 129)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

    train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
    test_dataset = TensorDataset(test_data_tensor, test_labels_tensor)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = NeuralNetwork(3)
    loss_fn = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)

    epochs = 50
    accuracies = {}

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

        accuracy = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {running_loss / len(train_loader):.4f} - Accuracy: {accuracy:.4f}")
        accuracies[epoch + 1] = accuracy
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=accuracies, marker='o')
    plt.title('Model Accuracy Over Training Epochs CNN+LSTM (OverSampled)', pad=15)
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1)  # Accuracy range is 0-1
    plt.tight_layout()
    plt.show()
    
    torch.save(model.state_dict(), 'models\\other_model_9.pth')

if __name__ == "__main__" :
    main()
 