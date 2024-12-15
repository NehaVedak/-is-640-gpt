import torch

class Trainer:
    def __init__(self, data, model, optimizer):
        self.data = data
        self.model = model
        self.optimizer = optimizer
        def train(self, iterations, batch_size, block_size):
        for iter in range(iterations):
