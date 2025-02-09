from pylsl import StreamInlet, resolve_stream
import numpy as np
import mne
import time
import tkinter

eeg_data = []

print(StreamInlet)

def get_data_from_stream(inlet: StreamInlet, chunk_size: int = 129) :
    samples, _ = inlet.pull_chunk(timeout=1.0, max_samples=chunk_size)
    if (len(samples) > 0) :
        eeg_data.extend(samples)
    return np.array(eeg_data)

print("Start")
streams = resolve_stream('type', 'EEG')
print("Resolved")
inlet = StreamInlet(streams[0])
sfreq = 129

if streams is not None :
    print(streams)

ch_names = ['Fp1.', 'Fp2.', 'F3..', 'F4..', 'C3..', 'C4..', 'F7..', 'F8..']
ch_types = ['eeg'] * 8

info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

while True :
    data = get_data_from_stream(inlet, chunk_size=129)
    print(data)
    print(data.shape)

