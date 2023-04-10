from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from load_data import ColumbiaPairs
from model.cross_encoder import CrossEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

eye_pairs = ColumbiaPairs('data')
generator = torch.Generator().manual_seed(42)
train_test = torch.utils.data.random_split(
    eye_pairs, [0.5, 0.5], generator=generator)

# TRAINING PARAMETERS
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.0001

# Create data loaders for our datasets; shuffle for training, not for validation
training_loader = torch.utils.data.DataLoader(
    train_test[0], batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    train_test[1], batch_size=BATCH_SIZE, shuffle=True)

model = CrossEncoder().to(device)

# loss function and optimizer
loss_fn = nn.L1Loss().to(device)  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train_one_epoch(epoch_index):
    running_loss = 0.
    num_batches = 0

    # EPOCH
    for i, data in tqdm(enumerate(training_loader)):
        # Zero gradients
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(data)

        # Compute the loss and its gradients
        loss = loss_fn(data, outputs.detach())
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        num_batches += 1

        if i % 100 == 99:
            last_loss = running_loss / 100  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            print('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


if __name__ == "__main__":
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    best_test_loss = 1_000_000.

    for epoch in range(EPOCHS):
        print('\nEPOCH {}:'.format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch)

        # We don't need gradients on to do reporting
        model.train(False)

        running_test_loss = 0.0
        for i, test_data in enumerate(test_loader):
            test_outputs = model(test_data)
            test_loss = loss_fn(test_data, test_outputs)
            running_test_loss += test_loss

        avg_test_loss = running_test_loss / (i + 1)
        print(f'LOSS train {avg_loss} valid {avg_test_loss}')
        print('Training vs. Test Loss',
              {'Training': avg_loss, 'Test': avg_test_loss},
              epoch + 1)

        # Track best performance, and save the model's state
        # TODO is this using test loss as validation loss?
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            model_path = 'models/model_{}_{}'.format(timestamp, epoch)
            torch.save(model.state_dict(), model_path)

        epoch += 1
