import torch.nn as nn
import torch.optim as optim
import torch
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from dataset import MatrixDataset
from dataset_transformer import MatrixDatasetTransformer
import pandas as pd
import os
import json

USER_PATH = os.environ["PROJECT"] + "/users/funk1/"
DATA_PATH = USER_PATH + 'data/'
DATASET_PATH = DATA_PATH + 'dataset.csv'
FEATURE_PATH = DATA_PATH + 'feature_vecs4096.json'

feature_vecs = json.load(open(FEATURE_PATH))
df = pd.read_csv(DATASET_PATH)
# df = df[df["path"].isin(list(feature_vecs.keys()))]
train_df = df.sample(frac=0.8)
test_df = df.drop(train_df.index)


train_set = MatrixDatasetTransformer(train_df)
test_set = MatrixDatasetTransformer(test_df)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=1,
                                           shuffle=True)
val_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                         shuffle=False)
device = torch.device("cuda:0")
print("device_count: ", torch.cuda.device_count())
print("device: ", torch.cuda.get_device_name(0))


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x


encoder_layer = nn.TransformerEncoderLayer(d_model=None, nhead=None)
model = nn.Sequential(
    nn.TransformerEncoder(encoder_layer, num_layers=6),
    PrintLayer(),
    nn.Flatten(),
    PrintLayer(),
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Linear(512, 7),
).to(device)

model.double()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

val_metrics = {
    "accuracy": Accuracy(),
    "nll": Loss(criterion)
}
evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)


@trainer.on(Events.ITERATION_COMPLETED(every=100))
def log_training_loss(trainer):
    print(f"Epoch[{trainer.state.epoch}] Loss: {trainer.state.output:.2f}")


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    print(
        f"Epoch: {trainer.state.epoch} accuracy: {metrics['accuracy']:.3f}, loss: {metrics['nll']:.3f}, ", end='')


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    print(
        f"val_accuracy: {metrics['accuracy']:.3f}, val_loss: {metrics['nll']:.3f}")


if __name__ == "__main__":
    trainer.run(train_loader, max_epochs=1000)
