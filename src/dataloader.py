import os
import torch
import numpy as np
import tiktoken
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from multiprocessing import cpu_count

tknzr = tiktoken.get_encoding("gpt2")
VOCAB_SIZE = tknzr.n_vocab
DATASET_DIR = "./datasets"

class TextDataset(Dataset):
    def __init__(self, data, sequence_length):
        super().__init__()
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        total_length = len(self.data)
        return (total_length - 1) // self.sequence_length

    def __getitem__(self, idx):
        seq_length = self.sequence_length
        idx = idx * seq_length
        x = torch.from_numpy((self.data[idx : idx + seq_length]).astype(np.int64))

        y = torch.from_numpy(
            (self.data[idx + 1 : idx + 1 + seq_length]).astype(np.int64)
        )
        return x, y

def get_data_loaders(batch_size=32, seq_len=128, dataset_dir=DATASET_DIR, num_proc=cpu_count()):
    """
    Downloads, tokenizes in parallel, saves to .bin files, and returns 
    PyTorch DataLoaders and metadata.
    """
    os.makedirs(dataset_dir, exist_ok=True)
    
    train_bin_path = os.path.join(dataset_dir, "train.bin")
    val_bin_path = os.path.join(dataset_dir, "val.bin")

    if not os.path.exists(train_bin_path):
        print("Binary files not found. Starting parallel tokenization and storage...")
        
        dataset = load_dataset("DKYoon/SlimPajama-6B")
        split_dataset = dataset["train"].train_test_split(
            test_size=0.0005, seed=2357, shuffle=True
        )
        split_dataset["val"] = split_dataset.pop("test")

        def process(example):
            ids = tknzr.encode_ordinary(example["text"])
            ids.append(tknzr.eot_token)
            out = {"ids": ids, "len": len(ids)}
            return out

        tokenized = split_dataset.map(
            process,
            remove_columns=["text"],
            desc="Tokenizing splits",
            num_proc=num_proc,
        )

        for split, dset in tokenized.items():
            print(f"\nWriting {split} data to binary file...")
            arr_len = np.sum(dset["len"])
            filename = os.path.join(dataset_dir, f"{split}.bin")
            dtype = np.uint16
            
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
            
            total_batches = min(1024, len(dset))

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"Writing {filename}"):
                batch = dset.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                ).with_format("numpy")
                arr_batch = np.concatenate(batch["ids"])
                
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            
            arr.flush()
            print(f"Successfully wrote {idx:,} tokens to {filename}")

    print("\nLoading data from memory-mapped binary files...")
    
    train_data = np.memmap(train_bin_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_bin_path, dtype=np.uint16, mode="r")

    train_ds = TextDataset(train_data, seq_len)
    val_ds = TextDataset(val_data, seq_len)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)

    meta = {
        "vocab_size": VOCAB_SIZE,
        "tokenizer": tknzr,
    }
    
    print(f"Train samples: {len(train_ds):,}, Val samples: {len(val_ds):,}")
    print("Data loaders initialized.")
    return train_loader, val_loader, meta
