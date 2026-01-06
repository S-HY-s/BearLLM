from dotenv import dotenv_values
import sqlite3
import random
import json
import h5pickle as h5py
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np

env = dotenv_values()
dataset_path = env['MBHM_DATASET']
data_path = f'{dataset_path}/data.hdf5'
meta_path = f'{dataset_path}/metadata.sqlite'
cache_path = f'{dataset_path}/dataset.json'
corpus_path = f'{dataset_path}/corpus.json'

def get_ref_ids():
    conn = sqlite3.connect(meta_path)
    cursor = conn.cursor()
    cursor.execute('SELECT condition_id, file_id FROM file_info WHERE label = 0')
    ref_data = cursor.fetchall()
    conn.close()
    ref_ids = {}
    for condition_id, file_id in ref_data:
        if condition_id not in ref_ids:
            ref_ids[condition_id] = []
        ref_ids[condition_id].append(file_id)
    return ref_ids


def get_condition_map():
    conn = sqlite3.connect(meta_path)
    cursor = conn.cursor()
    cursor.execute('SELECT file_id, condition_id FROM file_info')
    rows = cursor.fetchall()
    conn.close()
    return {file_id: condition_id for file_id, condition_id in rows}

def create_cache_dataset():
    ref_ids = get_ref_ids()
    conn = sqlite3.connect(meta_path)
    cursor = conn.cursor()
    cursor.execute('SELECT condition_id, file_id, label FROM file_info')
    data = cursor.fetchall()
    conn.close()

    dataset_info = {}
    for subset in ['train', 'val', 'test']:
        dataset_info[subset] = []

    now_num = 0
    for condition_id, file_id, label in data:
        if condition_id in ref_ids:
            c_ids = ref_ids[condition_id]
            r_ids = []
            for i in range(3):
                r_ids.append(random.choice(c_ids))
            if now_num % 10 < 7:
                subset = 'train'
            elif now_num % 10 < 9:
                subset = 'val'
            else:
                subset = 'test'
            for r_id in r_ids:
                dataset_info[subset].append([file_id, r_id, label])
            now_num += 1

    with open(cache_path, 'w') as f:
        json.dump(dataset_info, f)


def load_cache_dataset():
    if not os.path.exists(cache_path):
        create_cache_dataset()
    with open(cache_path, 'r') as f:
        dataset_info = json.load(f)
    return dataset_info

class VibDataset(Dataset):
    def __init__(self, subset_info):
        self.subset_info = subset_info
        self.data = h5py.File(data_path, 'r')['vibration']

    def __len__(self):
        return len(self.subset_info)

    def __getitem__(self, idx):
        file_id, ref_id, label = self.subset_info[idx]
        data = self.data[file_id]
        ref = self.data[ref_id]
        data = np.array([data, ref])
        return data, label


def get_datasets():
    dataset_info = load_cache_dataset()
    train_dataset = VibDataset(dataset_info['train'])
    val_dataset = VibDataset(dataset_info['val'])
    test_dataset = VibDataset(dataset_info['test'])
    return train_dataset, val_dataset, test_dataset

def get_loaders(batch_size, num_workers):
    train_set, val_set, test_set = get_datasets()
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=False)
    return train_loader, val_loader, test_loader


class CorpusDataset:
    def __init__(self):
        self.vib_data = h5py.File(data_path, 'r')['vibration']
        self.corpus = json.load(open(corpus_path, 'r'))
        self.condition_map = get_condition_map()
        self.healthy_indices_by_condition = get_ref_ids()

    def __len__(self):
        return len(self.corpus)

    def get_signal_item(self, idx):
        corpus_data = self.corpus[idx]
        sample_id = corpus_data['id']
        instruction = corpus_data['instruction']
        vib_id = corpus_data['vib_id']
        vib_data = self.vib_data[vib_id]
        condition_id = self.condition_map.get(vib_id)
        ref_index_pool = self.healthy_indices_by_condition.get(condition_id, [])
        label_id = corpus_data['label_id']
        return {
            'instruction': instruction,
            'vib_data': vib_data,
            'ref_index_pool': ref_index_pool,
            'condition': condition_id,
            'label': label_id,
            'sample_id': sample_id,
        }

    def get_text_item(self, idx):
        corpus_data = self.corpus[idx]
        return {
            'sample_id': corpus_data['id'],
            'instruction': corpus_data['instruction'],
            'response': corpus_data['response'],
            'label': corpus_data['label_id'],
        }

    def __getitem__(self, idx):
        return self.get_signal_item(idx)


if __name__ == '__main__':
    a, b, c = get_loaders(100, 10)
    print(len(a))
    for sample in a:
        print(sample[0].shape, sample[1].shape)
        print(sample[0].max(), sample[0].min())
        break
