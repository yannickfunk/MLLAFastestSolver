import os
import pandas as pd
from propulate import Propulator
from propulate.utils import get_default_propagator
from dataset import MatrixDataset
import torch
from torch import nn
from torch.utils.data import DataLoader

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

num_generations = 100
GPUS_PER_NODE = 4

limits = {
        'hiddenlayers': (1, 3),
        'neurons_1st': (16, 3000),
        'neurons_2nd': (1000, 7000),
        'neurons_3rd': (1000, 7000),
        'activation': ('relu', 'tanh'),
        'lr': (0.2, 0.00001),
        'batch_size': (400, 800)
        }


class Net(nn.Module):
    def __init__(self, hiddenlayers, neurons_hidden, activation):
        super(Net, self).__init__()
        flatten = lambda t: [item for sublist in t for item in sublist]
        layers = []
        layers += [nn.BatchNorm1d(18), nn.Linear(18, neurons_hidden[0]), activation()]
        layers += flatten([[nn.Linear(neurons_hidden[i], neurons_hidden[i+1]), activation()] for i in range(hiddenlayers-1)])
        layers += [nn.Linear(neurons_hidden[-1], 7)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def get_data_loaders(batch_size):
    USER_PATH = os.environ["PROJECT"] + "/users/funk1/"
    DATA_PATH = USER_PATH + 'data/'
    DATASET_PATH = DATA_PATH + 'dataset.csv'
    df = pd.read_csv(DATASET_PATH).dropna()
    train_df = df.sample(frac=0.8)
    test_df = df.drop(train_df.index)
    # print("train len: ", len(train_df))
    # print("test len: ", len(test_df))

    train_set = MatrixDataset(train_df)
    test_set = MatrixDataset(test_df)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                             shuffle=False, pin_memory=True)

    return train_loader, val_loader


def ind_loss(params):

    from mpi4py import MPI
    activations = {
        'relu': nn.ReLU,
        'sigmoid': nn.Sigmoid,
        'tanh': nn.Tanh
    }
    hiddenlayers = params['hiddenlayers']
    neurons_hidden = [
        params['neurons_1st'],
        params['neurons_2nd'],
        params['neurons_3rd'],
    ][:hiddenlayers]
    activation = activations[params['activation']]
    lr = params['lr']
    batch_size = params['batch_size']
    epochs = 50

    rank = MPI.COMM_WORLD.Get_rank()

    device = "cuda:{}".format(rank % GPUS_PER_NODE)

    model = Net(hiddenlayers, neurons_hidden, activation)
    model.to(device)
    model.double()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_loader, val_loader = get_data_loaders(batch_size)

    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    evaluator = create_supervised_evaluator(model, metrics={'acc': Accuracy(), 'ce': Loss(loss_fn)}, device=device)

    trainer.best_loss = 100000.0

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        # print(f"Epoch: {trainer.state.epoch} accuracy: {metrics['acc']:.3f}, loss: {metrics['ce']:.3f}, ", end='')

    @trainer.on(Events.EPOCH_COMPLETED)
    def validate(trainer):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        # print(f"val_accuracy: {metrics['acc']:.3f}, val_loss: {metrics['ce']:.3f}")
        if metrics['ce'] < trainer.best_loss:
            trainer.best_loss = metrics['ce']
    trainer.run(train_loader, max_epochs=epochs)
    print(f"Best loss {trainer.best_loss} with {params}")
    return trainer.best_loss


propagator = get_default_propagator(2, limits, .7, .4, .1)
propulator = Propulator(ind_loss, propagator, generations=num_generations, checkpoint_file="test.pkl")
propulator.propulate()
propulator.summarize(out_file="op_result.png")
