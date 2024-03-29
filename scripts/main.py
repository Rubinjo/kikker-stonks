from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch
import torch.nn as nn
import time
from sklearn.metrics import accuracy_score
import math
import os
import matplotlib.pyplot as plt

from cleaning import cleanData, splitingData
from neuralNet import LSTM_NN

# cleanData()

TEST_SIZE = 0.15
BATCH_SIZE = 50
LEARNING_RATE = 0.0003
NUM_EPOCHS = 150
SEQUENCE_LENGTH = 32


data = splitingData(SEQUENCE_LENGTH)
dataLength = math.floor(len(data) / 3)
training_size = math.floor(len(data) * (1 - TEST_SIZE) / 3)

train_encoded = data[0:training_size]
for item in data[dataLength : dataLength + training_size]:
    train_encoded.append(item)
for item in data[dataLength * 2 : dataLength * 2 + training_size]:
    train_encoded.append(item)

test_encoded = data[training_size:dataLength]
for item in data[(dataLength + training_size) : dataLength * 2]:
    test_encoded.append(item)
for item in data[(dataLength * 2 + training_size) : len(data)]:
    test_encoded.append(item)

# test_encoded, train_encoded = train_test_split(splitingData(SEQUENCE_LENGTH), test_size=(1 - TEST_SIZE), shuffle=False)
# train_encoded, test_encoded = train_test_split(splitingData(SEQUENCE_LENGTH), test_size=TEST_SIZE)

train_x = np.array([ticker for ticker, label in train_encoded], dtype=np.float32)
train_y = np.array([label for ticker, label in train_encoded], dtype=np.float32)
test_x = np.array([ticker for ticker, label in test_encoded], dtype=np.float32)
test_y = np.array([label for ticker, label in test_encoded], dtype=np.float32)

train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
test_ds = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

train_dl = DataLoader(train_ds, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
test_dl = DataLoader(test_ds, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTM_NN(BATCH_SIZE, SEQUENCE_LENGTH, 15, 1, 0.2)
# model = model.to(dtype=torch.double)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

start = time.time()
train_losses = []
test_losses = []

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start)))
for epoch in range(NUM_EPOCHS):
    h0, c0 = model.init_hidden()

    h0 = h0.to(device)
    c0 = c0.to(device)

    # Train mode
    model.train()

    for batch_idx, batch in enumerate(train_dl):

        input, target = batch[0].to(device), batch[1].to(device)
        # print(input)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            # print(input, h0, c0)
            out, hidden = model(input, (h0, c0))
            # print(out)
            train_loss = criterion(out, target.long())
            # print(train_loss)
            train_loss.backward()
            optimizer.step()

        pred = torch.argmax(out, dim=1)
        correct = torch.sum(torch.eq(pred, target)).item()

        elapsed = time.time() - start

        if not batch_idx % (math.ceil(len(train_dl) / 4)):
            print(
                f"epoch: {epoch}, batch: {batch_idx:<{len(str(len(train_dl)))}}/{len(train_dl)}, time: {elapsed:.3f}s, loss: {train_loss.item():.3f}, acc: {correct / BATCH_SIZE:.3f}"
            )

    train_losses.append(train_loss.item())

    # Evaluation mode
    model.eval()

    batch_acc = []
    for batch_idx, batch in enumerate(test_dl):

        input, target = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(False):
            out, hidden = model(input, (h0, c0))
            _, preds = torch.max(out, 1)
            preds = preds.to(device).tolist()
            batch_acc.append(accuracy_score(preds, target.tolist()))

            # print(preds, target)
            test_loss = criterion(out, target.long())

    print(f"Accuracy on the test set: {sum(batch_acc)/len(batch_acc):.3f}")

    test_losses.append(test_loss.item())


print("total time: ", time.time() - start)
print("time cost: ", time.strftime("%H:%M:%S", time.gmtime(time.time() - start)))

path = "model/"
if not os.path.exists(path):
    os.makedirs(path)

curTime = time.time()

torch.save(model.state_dict(), path + f"model{curTime}.pth")

# loading model:
# model = LSTM_NN(BATCH_SIZE, 32, 16, 4, 0.2)
# model.load_state_dict(torch.load("model/model39p.pth"))
# model.eval()

fig = plt.figure()

plt.plot(train_losses, label="Training loss")
plt.plot(test_losses, label="Validation loss")
plt.title("Training and Validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
fig.savefig("train-val-loss.png")
