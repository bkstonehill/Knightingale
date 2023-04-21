import time
from datetime import datetime

import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from BRVIT import BRViT
from Knightingale import fen2img

np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

def collate_fn(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.FloatTensor(target)
    return [data, target]


class ChessDataset(Dataset):
    def __init__(self, csv_file, transform=None, target_transform=None):
        self.data = pd.read_csv(csv_file, index_col=0)
        self.data['Game Sequence'] = self.data['Game Sequence'].apply(lambda x: np.array(eval(x)), 0)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        position_sequence = self.data.iloc[item, 0]
        score = self.data.iloc[item, 1]
        if self.transform:
            position_sequence = self.transform(position_sequence)
        if self.target_transform:
            score = self.target_transform(score)
        return position_sequence, score


def train(training_data: DataLoader, validation_data: DataLoader, model: BRViT,
          loss_fn, batch_size=100, optimzer=torch.optim.NAdam, epochs=100,
          save_path='./chessmodel'):
    # Take in epochs, training data, validation data, loss fn, optimizer, model
    # weight save location

    # Initialize tensorboard writer
    # loop through number of epochs
    # In each epoch:
    #   set training = True
    #   for batch in training data:
    #       split to data and label
    #       zero gradients
    #       predict values
    #       compute loss
    #       back propagation
    #       optimizer step for weight adjustments
    #   compute avg loss for epoch
    #   set training = False
    #   for batch in validation data:
    #       split to data and label
    #       predict values
    #       compute loss
    #   compute avg loss for epoch
    #   log training and validation loss to tensorboard writer
    #   save current weights if avg validation loss is smaller than the best validation loss

    optim = optimzer(model.parameters())
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'./runs/chess_trainer_{timestamp}')
    best_vloss = math.inf
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # For epoch
    print('Starting Training')
    for epoch in range(epochs):
        print(f'Starting epoch: {epoch+1}')
        epoch_losses_training = list()
        epoch_losses_validation = list()

        model.train(True)

        # For batch in training
        for i, data in enumerate(training_data):
            # Split data to inputs and labels
            inputs, labels = data
            hidden = torch.zeros(batch_size, 4, 256)

            inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True)

            # Zero gradients
            optim.zero_grad()

            inputs, labels, hidden = inputs.to(device), labels.to(device), hidden.to(device)
            for j in range(inputs.shape[1]):
                out, hidden = model(inputs[:, j], hidden)

            # Compute loss and backprogation for each sequence
            loss = loss_fn(out, labels.reshape(batch_size, 1))
            loss.backward()

            # Optimizer step
            optim.step()

            print(f'     batch {i}: loss: {loss.item()}')

            # Track loss per batch
            epoch_losses_training.append(loss.detach().item())

        # Compute avg training loss
        training_loss = torch.tensor(epoch_losses_training).mean()

        model.train(False)

        torch.cuda.empty_cache()

        # For batch in validation
        for i, data in enumerate(validation_data):
            # Split data to inputs and labels
            inputs, labels = data
            hidden = torch.zeros(labels.shape[0], 4, 256)

            inputs = nn.utils.rnn.pad_sequence(inputs, batch_first=True)

            inputs, labels, hidden = inputs.to(device), labels.to(device), hidden.to(device)

            for j in range(inputs.shape[1]):
                out, hidden = model(inputs[:, j], hidden)

            # Compute loss for each sequence
            loss = loss_fn(out, labels.reshape(labels.shape[0], 1))

            # Track loss per batch
            epoch_losses_validation.append(loss.detach().item())

        validation_loss = torch.tensor(epoch_losses_validation).mean()

        print(f'Epoch: {epoch + 1}, loss: {training_loss}, val_loss: {validation_loss}')
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': training_loss, 'Validation': validation_loss},
                           epoch + 1)
        writer.flush()

        if validation_loss < best_vloss:
            best_vloss = validation_loss
            torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    # Load chess games (stored as list of fen strings for positions) into DataFrame
    # Convert fen strings to images
    # Load into PyTorch data loader
    # Pass to training function
    batch_size = 50
    training_data = ChessDataset('./ChessDatabase/Database-10k-training.csv',
                                 transform=lambda x: torch.stack([fen2img(position) for position in x]),
                                 target_transform=lambda x: eval(x))

    validation_data = ChessDataset('./ChessDatabase/Database-10k-testing.csv',
                                   transform=lambda x: torch.stack([fen2img(position) for position in x]),
                                   target_transform=lambda x: eval(x))

    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    validation_loader = DataLoader(validation_data, batch_size=10, shuffle=False, collate_fn=collate_fn)

    model = BRViT(patch_size=4, din=(100, 1, 8, 8),
                  dmodel=256, dff=1024, nheads=4,
                  nlayers=6, dout=1, out_activation=None, dropout=0.1)

    if torch.cuda.is_available():
        train(train_loader, validation_loader, model, loss_fn=nn.MSELoss(), batch_size=batch_size)
    else:
        print("GPU IS NOT AVAILABLE")