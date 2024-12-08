import torch

class Data:
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)

        # Create mappings
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

        # Encode text to integers
        self.data = torch.tensor([self.stoi[c] for c in self.text], dtype=torch.long)

    def encode(self, text):
        """Convert string to list of integers"""
        return [self.stoi[ch] for ch in text]

    def decode(self, indices):
        """Convert list of integers to string"""
        return ''.join([self.itos[i] for i in indices])

def get_splits(self, split_ratio=0.9):
        """Split data into training and validation sets"""
        n = int(len(self.data) * split_ratio)
        return self.data[:n], self.data[n:]
