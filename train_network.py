import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import load_data
import gaze_estimator

data_set = load_data.create_dataset_from_mix("")
generator = torch.Generator().manual_seed(42)
train_test = torch.utils.data.random_split(
    data_set, [0.5, 0.5], generator=generator)

# Create data loaders for our datasets; shuffle for training, not for validation
training_loader = torch.utils.data.DataLoader(
    train_test[0], batch_size=10, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    train_test[1], batch_size=10, shuffle=True)

model = gaze_estimator.SupervisedGazeEstimator()

# loss function and optimizer
loss_fn = nn.L1Loss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_one_epoch(epoch_index):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            print('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


def main():
    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    EPOCHS = 5

    best_test_loss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch)

        # We don't need gradients on to do reporting
        model.train(False)

        running_test_loss = 0.0
        for i, test_data in enumerate(test_loader):
            test_inputs, test_labels = test_data
            test_outputs = model(test_inputs)
            test_loss = loss_fn(test_outputs, test_labels)
            running_test_loss += test_loss

        avg_test_loss = running_test_loss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_test_loss))

        # Log the running loss averaged per batch
        # for both training and validation
        print('Training vs. Validation Loss',
              {'Training': avg_loss, 'Validation': avg_test_loss},
              epoch + 1)

        # Track best performance, and save the model's state
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            model_path = 'model_{}_{}'.format(timestamp, epoch)
            torch.save(model.state_dict(), model_path)

        epoch += 1


if __name__ == "__main__":
    main()
