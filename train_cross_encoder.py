from datetime import datetime
import loss
from torch.nn import L1Loss
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm
from load_data import ColumbiaPairs, NUM_SUBJECTS
from model.cross_encoder import CrossEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

# TRAINING PARAMETERS
BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 0.0001
TRAIN_TEST_SPLIT = [0.8, 0.2]
ESTIMATOR_EPOCHS = 30
ESTIMATOR_BATCH_SIZE = 16
ESTIMATOR_LEARNING_RATE = 0.0001

generator = torch.Generator().manual_seed(42)
train_subjects, test_subjects = random_split(
    range(NUM_SUBJECTS), TRAIN_TEST_SPLIT, generator=generator)

train_set = ColumbiaPairs('data', train_subjects)
test_set = ColumbiaPairs('data', test_subjects)

training_loader = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(
    test_set, batch_size=BATCH_SIZE, shuffle=True)

model = CrossEncoder().to(device)

# loss function and optimizer
estimator_loss = L1Loss()
loss_fn = loss.loss_fn
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def reshape_for_estimator(outputs, labels):
    # repeat label for each gaze pair
    labels = labels.repeat_interleave(2, dim=0)

    # take only gaze pair output (first two images)
    outputs = outputs[:, :2, :]
    outputs = outputs.reshape(-1, 2)

    # labels.shape = (32, 2)
    # outputs.shape = (32, 2)

    return outputs, labels


def train_one_epoch(estimator=False):
    '''
    Train the model on one full epoch.
    estimator : bool
        trains the model in estimator mode or cross-encoder mode
    '''
    running_loss = 0.

    # EPOCH
    for data, labels in tqdm(training_loader, desc="Training..."):
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

    return running_loss / float(len(training_loader))


if __name__ == "__main__":
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    best_test_loss = 1_000_000.

    for epoch in range(EPOCHS):
        print(f'\nEPOCH {epoch + 1} / {EPOCHS}')

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch()

        # We don't need gradients on to do reporting
        model.train(False)

        running_test_loss = 0.0
        for test_data, _ in tqdm(test_loader, desc="Testing..."):

            test_data = test_data.to(device)

            test_outputs = model(test_data)
            test_loss = loss_fn(test_data, test_outputs)

            running_test_loss += test_loss.item()

        avg_test_loss = running_test_loss / len(test_data)
        print(f'LOSS train {avg_loss} valid {avg_test_loss}')
        print('Training vs. Test Loss',
              {'Training': avg_loss, 'Test': avg_test_loss, 'Epoch': epoch + 1})

        # Track best performance, and save the model's state
        torch.save(model.state_dict(), f'models/model_{timestamp}_{epoch}')

    # Estimator training
    for epoch in range(ESTIMATOR_EPOCHS):
        print(f'\nEPOCH {epoch + 1} / {EPOCHS}')

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(estimator=True)

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

        avg_test_loss = running_test_loss / len(test_data)
        print(f'LOSS train {avg_loss} valid {avg_test_loss}')
        print('Training vs. Test Loss',
              {'Training': avg_loss, 'Test': avg_test_loss, 'Epoch': epoch + 1})

        torch.save(model.state_dict(), f'models/model_est_{timestamp}_{epoch}')
