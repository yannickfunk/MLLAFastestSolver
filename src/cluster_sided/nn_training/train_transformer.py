import torch.nn as nn
import torch.optim as optim
import torch
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from dataset_transformer import MatrixDatasetTransformer
import pandas as pd
import os
from linear_attention_transformer import LinearAttentionTransformer

USER_PATH = os.environ["PROJECT"] + "/users/funk1/"
DATA_PATH = USER_PATH + 'data/'
DATASET_PATH = DATA_PATH + 'dataset.csv'

df = pd.read_csv(DATASET_PATH).dropna()
df = df[df["nonzeros"] < 100000]
train_df = df.sample(frac=0.8)
test_df = df.drop(train_df.index)
print("train len: ", len(train_df))
print("test len: ", len(test_df))
test_df.to_csv("val_set.csv", index=False)
train_set = MatrixDatasetTransformer(train_df)
test_set = MatrixDatasetTransformer(test_df)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=1,
                                           shuffle=True, pin_memory=True)
val_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                         shuffle=False, pin_memory=True)
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


class Average(nn.Module):
    def __init__(self):
        super(Average, self).__init__()

    def forward(self, x):
        return x.mean(dim=1)


model = nn.Sequential(
    LinearAttentionTransformer(
        dim=3,
        heads=3,
        depth=1,
        max_seq_len=9895422,
        n_local_attn_heads=2
    ),
    Average(),

    nn.Linear(3, 7),
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
trainer.best_accuracy = 0.0


@trainer.on(Events.ITERATION_COMPLETED(every=1000))
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
    """
    if metrics['accuracy'] > trainer.best_accuracy:
        trainer.best_accuracy = metrics['accuracy']
        filename = f"models/model_{metrics['nll']:.2f}_{metrics['accuracy']:.4f}.pt"
        torch.save(model.state_dict(), filename)
        print(f"saved {filename}")
    """


if __name__ == "__main__":
    trainer.run(train_loader, max_epochs=10000)