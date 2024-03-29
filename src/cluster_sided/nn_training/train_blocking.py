import torch.nn as nn
import torch.optim as optim
import torch
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from dataset_blocking import MatrixDatasetBlocking
import pandas as pd
import os
import json

from Informer2020.models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from Informer2020.models.attn import FullAttention, ProbAttention, AttentionLayer
from Informer2020.models.embed import DataEmbedding

USER_PATH = os.environ["PROJECT"] + "/users/funk1/"
DATA_PATH = USER_PATH + 'data/'
DATASET_PATH = DATA_PATH + 'dataset.csv'
FEATURE_PATH = DATA_PATH + 'blocking_vec.json'

df = pd.read_csv(DATASET_PATH).dropna()
train_df = df.sample(frac=0.8)
test_df = df.drop(train_df.index)
print("train len: ", len(train_df))
print("test len: ", len(test_df))
test_df.to_csv("val_set.csv", index=False)
train_set = MatrixDatasetBlocking(train_df, FEATURE_PATH)
test_set = MatrixDatasetBlocking(test_df, FEATURE_PATH)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=1,
                                           shuffle=True, pin_memory=True)
val_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                         shuffle=False, pin_memory=True)
device = torch.device("cuda:2")
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


class InformerEnc(nn.Module):
    def __init__(self, factor=5, d_model=11, n_heads=8, e_layers=5, d_ff=512, dropout=0.0, activation='gelu',
                 output_attention=False, distil=True):
        super(InformerEnc, self).__init__()

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(ProbAttention(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for _ in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )

    def forward(self, x, enc_self_mask=None):
        enc_out, attns = self.encoder(x, attn_mask=enc_self_mask)
        return enc_out


model = nn.Sequential(
    InformerEnc(d_model=11),
    Average(),
    nn.Linear(11, 447),
    nn.ReLU(),

    nn.Linear(447, 1324),
    nn.ReLU(),

    nn.Linear(1324, 7),
).to(device)

model.double()
optimizer = optim.Adam(model.parameters(), lr=0.001)
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