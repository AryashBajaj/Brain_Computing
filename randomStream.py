import argparse
import csv
import os
import random
import time
from typing import List

import numpy as np
import pylsl


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sample_rate', type=int, required=True,
                    help='frequency of stream in Hz')
parser.add_argument('-f', '--eeg_csv_path', type=str, required=False,
                    help='path to EEG CSV file to stream')
args = parser.parse_args()
if args.eeg_csv_path:
    if not os.path.isfile(args.eeg_csv_path):
        raise FileNotFoundError(f'could not find csv file {args.eeg_csv_path}')
    if not args.eeg_csv_path.endswith('.csv'):
        raise ValueError(f'Expected .csv extension on file {args.eeg_csv_path}')
if args.sample_rate < 0:
    raise ValueError(f'Invalid sample rate, must be >= 0: {args.sample_rate}')

info = pylsl.StreamInfo(
    'SimulatedEEGStream',
    'EEG',
    8,
    args.sample_rate,
    'float32',
    'muse'
)
outlet = pylsl.StreamOutlet(info)
start_ts = pylsl.local_clock()
sample_id = 0


def stream_sample(eeg_sample: List[float]):
    global sample_id
    ideal_ts = start_ts + (1 / args.sample_rate * sample_id)
    outlet.push_sample(sample)
    time_diff_secs = pylsl.local_clock() - ideal_ts
    if time_diff_secs < 0:
        time.sleep(-time_diff_secs)
    sample_id += 1


if args.eeg_csv_path:
    with open(args.eeg_csv_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)
        for row in csv_reader:
            sample = [int(row[channel_id]) for channel_id in range(8)]
            stream_sample(eeg_sample=sample)
else:
    while True:
        sample = [random.random() for _ in range(8)]
        stream_sample(eeg_sample=sample)

print('Done.')