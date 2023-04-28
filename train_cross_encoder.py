from datetime import datetime
import loss
from torch.nn import L1Loss
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm
from load_data import ColumbiaPairs, NUM_SUBJECTS
from model.cross_encoder import CrossEncoder
from itertools import islice
from angular_error import compute_angular_error

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

# TRAINING PARAMETERS
BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 1e-4
TRAIN_TEST_SPLIT = [0.8, 0.2]
ESTIMATOR_EPOCHS = 90
ESTIMATOR_BATCH_SIZE = 8
ESTIMATOR_LEARNING_RATE = 1e-3

# split data
generator = torch.Generator().manual_seed(42)
train_subjects, test_subjects = random_split(
    range(NUM_SUBJECTS), TRAIN_TEST_SPLIT, generator=generator)

train_set = ColumbiaPairs('data', train_subjects)
test_set = ColumbiaPairs('data', test_subjects)

training_loader = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(
    test_set, batch_size=BATCH_SIZE, shuffle=False)

# extract 100 shots for estimator training
estimator_data = list(islice(train_set, 100))
estimator_training_loader = DataLoader(
    estimator_data, batch_size=ESTIMATOR_BATCH_SIZE, shuffle=True)

# initialize model
model = CrossEncoder().to(device)

# loss function and optimizer
estimator_loss = L1Loss()
loss_fn = loss.loss_fn
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
estimator_optimizer = optim.Adam(
    model.parameters(), lr=ESTIMATOR_LEARNING_RATE)


def reshape_for_estimator(outputs, labels):
    # repeat label for each gaze pair
    labels = labels.repeat_interleave(2, dim=0)

    # take only gaze pair output (first two images)
    outputs = outputs[:, :2, :]
    outputs = outputs.reshape(-1, 2)

    # labels.shape = (32, 2)
    # outputs.shape = (32, 2)

    return outputs, labels


def train_one_epoch(data_loader, estimator=False):
    '''
    Train the model on one full epoch.
    estimator : bool
        trains the model in estimator mode or cross-encoder mode
    '''
    running_loss = 0.

    # EPOCH
    for data, labels in tqdm(data_loader, desc="Training..."):
        # Zero gradients
        optimizer.zero_grad()

        data = data.to(device)
        labels = labels.to(device)

        # Make predictions for this batch
        outputs = model(data, estimator)

        # cross-encoder mode
        # data.shape = (64, 1, 32, 64)
        # outputs.shape = (64, 1, 32, 64)

        # estimator mode
        # data.shape = (64, 1, 32, 64)
        # outputs.shape = (16, 4, 2)

        # Compute the loss and its gradients
        if estimator:
            outputs, labels = reshape_for_estimator(outputs, labels)
            loss = estimator_loss(outputs, labels)
        else:
            loss = loss_fn(data, outputs)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

    return running_loss / len(data_loader.dataset)


if __name__ == "__main__":
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    best_test_loss = 1_000_000.

    # training encoder-decoder
    for epoch in range(EPOCHS):
        print(f'\nEPOCH {epoch + 1} / {EPOCHS}')

        # loss is for 4 images per sample
        model.train(True)
        avg_loss = train_one_epoch(training_loader) / 4

        # We don't need gradients on to do reporting
        model.train(False)

        running_test_loss = 0.0
        for test_data, _ in tqdm(test_loader, desc="Testing..."):

            test_data = test_data.to(device)

            test_outputs = model(test_data)
            test_loss = loss_fn(test_data, test_outputs)

            running_test_loss += test_loss.item()

        avg_test_loss = running_test_loss / len(test_loader.dataset)
        print(f'LOSS train {avg_loss} valid {avg_test_loss}')
        print('Training vs. Test Loss',
              {'Training': avg_loss, 'Test': avg_test_loss, 'Epoch': epoch + 1})

        model_path = 'models/model_{}_{}'.format(timestamp, epoch)
        torch.save(model.state_dict(), model_path)

    # training encoder-estimator
    for epoch in range(ESTIMATOR_EPOCHS):
        print(f'\nEPOCH {epoch + 1} / {ESTIMATOR_EPOCHS}')

        # loss is for 2 images per sample
        model.train(True)
        avg_loss = train_one_epoch(
            estimator_training_loader, estimator=True) / 2

        # We don't need gradients on to do reporting
        model.train(False)

        running_test_loss = 0.0
        for test_data, labels in tqdm(test_loader, desc="Testing..."):

            test_data = test_data.to(device)
            labels = labels.to(device)

            test_outputs = model(test_data, estimator=True)

            test_outputs, labels = reshape_for_estimator(test_outputs, labels)
            test_loss = estimator_loss(test_outputs, labels)

            running_test_loss += test_loss.item()

        avg_test_loss = running_test_loss / len(test_loader.dataset) / 2
        print(f'LOSS train {avg_loss} valid {avg_test_loss}')
        print('Training vs. Test Loss',
              {'Training': avg_loss, 'Test': avg_test_loss, 'Epoch': epoch + 1})

        torch.save(model.state_dict(), f'models/model_est_{timestamp}_{epoch}')

    # calculate angular error
    model.train(False)

    running_angular_error = 0.0
    for test_data, labels in tqdm(test_loader, desc="Testing..."):

        test_data = test_data.to(device)
        labels = labels.to(device)

        test_outputs = model(test_data, estimator=True)

        test_outputs, labels = reshape_for_estimator(test_outputs, labels)
        test_error = compute_angular_error(test_outputs, labels)

        running_angular_error += test_error.item()

    print('\nangular error:')
    print(running_angular_error / len(test_loader.dataset) / 2)
