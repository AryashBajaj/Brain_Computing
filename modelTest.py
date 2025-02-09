import torch
import numpy as np
import random
from torch import nn
from sklearn.preprocessing import StandardScaler
from collections import Counter
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

model = NeuralNetwork(3)
model.load_state_dict(torch.load('models\\other_model_7.pth'))
model.eval()

test_data = np.load('Data_files\\eeg_dataset_data.npy')
test_labels = np.load('Data_files\\eeg_dataset_labels.npy') - 1
unique_labels = np.unique(test_labels)
unexpected_labels = unique_labels[unique_labels > 2]
print(unique_labels, unexpected_labels)

scaler = StandardScaler()
test_data_scaled = scaler.fit_transform(test_data.reshape(test_data.shape[0], -1))
test_data_scaled = test_data_scaled.reshape(test_data.shape)
print(test_data_scaled.shape)

test_data_tensor = torch.tensor(test_data_scaled, dtype=torch.float32).view(-1, 8, 129)
test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

label_counts = Counter(test_labels)
print("Label frequency in the dataset:")
for label, count in label_counts.items():
    print(f"Label {label}: {count} samples")

labels, counts = zip(*label_counts.items())
plt.bar(labels, counts)
plt.xlabel('Labels')
plt.ylabel('Frequency')
plt.title('Label Distribution in Dataset')
plt.show()

num_samples = 100
random_indices = random.sample(range(len(test_data_tensor)), num_samples)

correct_predictions = 0

# Extract the first instance of each class
first_instances = {}
for i in range(len(test_labels)):
    label = test_labels[i]
    if label not in first_instances:
        first_instances[label] = test_data_tensor[i]
    if len(first_instances) == 3:  # We only need one instance for each class
        break

# Plot the first instance for each class
fig, axes = plt.subplots(3, 8, figsize=(20, 12))  # 3 rows (classes), 8 columns (channels)
fig.suptitle('First Instance of Each Class (8-Channel EEG)', fontsize=16)

for class_idx, (label, eeg_data) in enumerate(first_instances.items()):
    for channel_idx in range(8):
        axes[class_idx, channel_idx].plot(eeg_data[channel_idx].numpy(), color='blue')
        axes[class_idx, channel_idx].set_title(f'Class {label} - Ch {channel_idx + 1}', fontsize=10)
        axes[class_idx, channel_idx].set_xticks([])
        axes[class_idx, channel_idx].set_yticks([])
        axes[class_idx, channel_idx].grid(True)
        
    # Add a label to the left side for each row
    axes[class_idx, 0].set_ylabel(f'Class {label}', fontsize=12, rotation=0, labelpad=40)

# Adjust spacing
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


