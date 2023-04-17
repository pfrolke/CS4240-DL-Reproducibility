from datetime import datetime
import loss
from torch.nn import L1Loss
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from load_data import ColumbiaPairs
from model.cross_encoder import CrossEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

# TRAINING PARAMETERS
BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 0.0001
TRAIN_TEST_SPLIT = [0.5, 0.5]

# Create data loaders for our datasets; shuffle for training, not for validation
eye_pairs = ColumbiaPairs('data')
generator = torch.Generator().manual_seed(42)
train_test = torch.utils.data.random_split(
    eye_pairs, TRAIN_TEST_SPLIT, generator=generator)

training_loader = DataLoader(
    train_test[0], batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(
    train_test[1], batch_size=BATCH_SIZE, shuffle=True)

model = CrossEncoder().to(device)

# loss function and optimizer
estimator_loss = L1Loss()
loss_fn = loss.loss_fn
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train_one_epoch(estimator=False):
    '''
    Train the model on one full epoch.
    estimator : bool
        trains the model in estimator mode or cross-encoder mode
    '''
    running_loss = 0.
    num_batches = 0

    # EPOCH
    for data, labels in tqdm(training_loader, desc="Training..."):
        # Zero gradients
        optimizer.zero_grad()

        data = data.to(device)

        # Make predictions for this batch
        outputs = model(data, estimator)

        # cross-encoder mode
        # data.shape = (64, 1, 32, 64)
        # outputs.shape = (64, 1, 32, 64)

        # estimator mode
        # data.shape = (64, 1, 32, 64)
        # outputs.shape = (64, 1, 2)

        # Compute the loss and its gradients
        if estimator:
            loss = estimator_loss(outputs, labels)
        else:
            loss = loss_fn(data, outputs)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        num_batches += 1

    return running_loss / float(num_batches)


if __name__ == "__main__":
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    best_test_loss = 1_000_000.

    for epoch in range(EPOCHS):
        print(f'\nEPOCH {epoch + 1} / {EPOCHS}')

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch)

        # We don't need gradients on to do reporting
        model.train(False)

        running_test_loss = 0.0
        for test_data, _ in tqdm(test_loader, desc="Testing..."):

            test_data = test_data.to(device)

            test_outputs = model(test_data)
            test_loss = loss_fn(test_data, test_outputs)

            running_test_loss += test_loss

        avg_test_loss = running_test_loss / len(test_data)
        print(f'LOSS train {avg_loss} valid {avg_test_loss}')
        print('Training vs. Test Loss',
              {'Training': avg_loss, 'Test': avg_test_loss, 'Epoch': epoch + 1})

        # Track best performance, and save the model's state
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            model_path = 'models/model_{}_{}'.format(timestamp, epoch)
            torch.save(model.state_dict(), model_path)
