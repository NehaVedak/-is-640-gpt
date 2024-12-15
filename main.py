import torch
from trainer import Trainer
from data import Data
from model import GPTLanguageModel

# Hyperparameters
random_seed = 1337
train_iterations = 100
word_count = 100
data_file = "input.txt"
block_size = 8
batch_size = 16
learning_rate = 0.001

def main():
    """Main function"""
    torch.manual_seed(random_seed)

    # Load data
    data = Data(data_file)
