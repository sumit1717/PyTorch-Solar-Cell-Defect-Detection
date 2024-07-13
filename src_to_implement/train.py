import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
data_frame = pd.read_csv('data.csv')
train_data, val_data = train_test_split(data_frame, test_size=0.2, random_state=42)

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
train_dataset = ChallengeDataset(train_data, mode='train')
val_dataset = ChallengeDataset(val_data, mode='val')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# create an instance of our ResNet model
resnet_model = model.ResNet(num_classes=2)

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion
criterion = t.nn.BCEWithLogitsLoss()
optimizer = t.optim.Adam(resnet_model.parameters(), lr=0.001)
trainer = Trainer(model=resnet_model, crit=criterion, optim=optimizer,
                  train_dl=train_loader, val_test_dl=val_loader, cuda=True, early_stopping_patience=5)

# go, go, go... call fit on trainer
res = trainer.fit(epochs=20)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')