import os
import mne
import numpy as np
import torch

def preprocess_eeg_file(edf_file):
    print(f"Processing file: {edf_file}")
    raw = mne.io.read_raw_edf(edf_file, preload=True)
    print(f"Loaded raw data: {raw}")
    
    events, events_id = mne.events_from_annotations(raw)
    print(f"Events found: {events.shape[0]}")
    
    raw.load_data()
    raw.filter(1., 40., fir_design='firwin')
    print("Data filtered between 1 and 40 Hz")
    
    epochs = mne.Epochs(raw, events, event_id=events_id, tmin=-0.2, tmax=0.8, baseline=(None, 0), preload=True)
    epochs.drop_bad()
    print(f"Epochs created and bad epochs dropped. Remaining epochs: {len(epochs)}")
    
    data = epochs.get_data()
    labels = epochs.events[:, -1]
    print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
    return data, labels

def process_subfolder(subfolder_path):
    print(f"Processing subfolder: {subfolder_path}")
    all_data = []
    all_labels = []
    
    for filename in os.listdir(subfolder_path):
        if filename.endswith('.edf'):
            edf_file = os.path.join(subfolder_path, filename)
            data, labels = preprocess_eeg_file(edf_file)
            if data.shape[-1] != 129: continue
            all_data.append(data)
            all_labels.append(labels)
    
    if len(all_data) == 0:
        return [], []
    print(f"Finished processing subfolder: {subfolder_path}")
    return np.concatenate(all_data, axis=0), np.concatenate(all_labels, axis=0)

main_folder = 'dataset\\files2'

dataset_data = []
dataset_labels = []

print(os.listdir(main_folder))
for subfolder in sorted(os.listdir(main_folder)):
    print("In the data folder")
    subfolder_path = os.path.join(main_folder, subfolder)
    if os.path.isdir(subfolder_path):
        print(f'Starting to process subfolder: {subfolder}')
        data, labels = process_subfolder(subfolder_path)
        if len(data) == 0 or len(labels) == 0 :
            continue
        dataset_data.append(data)
        dataset_labels.append(labels)
        print(f"Processed subfolder: {subfolder}")

dataset_data = np.concatenate(dataset_data, axis=0)
dataset_labels = np.concatenate(dataset_labels, axis=0)

print(f"Total dataset size: {dataset_data.shape}, Total labels size: {dataset_labels.shape}")

np.save('eeg_dataset_data.npy', dataset_data)
np.save('eeg_dataset_labels.npy', dataset_labels)

print(f"Dataset saved. Total samples: {len(dataset_labels)}")