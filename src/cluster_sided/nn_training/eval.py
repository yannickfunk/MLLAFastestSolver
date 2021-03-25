import pandas as pd
from dataset_eval import MatrixEvalDataset
import torch
from torch import nn
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
import numpy as np


def get_top_n_accuracy(n):
    for features, labels in val_loader:
        features = features.to(device)
        predictions = model(features).cpu().detach().numpy().argmax(axis=1)
        labels = labels.numpy()[:, :n]
        matches = len([pred for pred, top_n in zip(predictions, labels) if pred in top_n])
        accuracy = matches / len(eval_set)
        return accuracy


def get_top_n_slowdown(n):
    solver_results = np.array(df.iloc[:, 1:8])
    for features, labels in val_loader:
        features = features.to(device)
        predictions = model(features).cpu().detach().numpy().argmax(axis=1)
        fp = 0
        labels = labels.numpy()[:, :n]
        slowdowns = []
        for i, (pred, l) in enumerate(zip(predictions, labels)):
            if pred != l[0] and pred in l:
                fp += 1
                slowdown_factor = solver_results[i, pred] / solver_results[i, l[0]]
                slowdowns.append(slowdown_factor)
        slowdowns = np.array(slowdowns)
        return slowdowns


def get_slowdown():
    solver_results = np.array(df.iloc[:, 1:8])
    for features, labels in val_loader:
        features = features.to(device)
        predictions = model(features).cpu().detach().numpy().argmax(axis=1)
        fp = 0
        labels = labels.numpy()[:, 0]
        slowdowns = []
        for i, (pred, l) in enumerate(zip(predictions, labels)):
            if pred != l:
                fp += 1
                slowdown_factor = solver_results[i, pred] / solver_results[i, l]
                slowdowns.append(slowdown_factor)
        slowdowns = np.array(slowdowns)
        return slowdowns


df = pd.read_csv("val_set.csv")
eval_set = MatrixEvalDataset(df)
val_loader = torch.utils.data.DataLoader(eval_set, batch_size=len(eval_set),
                                         shuffle=False, pin_memory=True)
model_path = "models/model_1.28_0.6016.pt"
device = torch.device("cuda:2")

model = nn.Sequential(
    nn.BatchNorm1d(18),

    nn.Linear(18, 447),
    nn.ReLU(),

    nn.Linear(447, 1324),
    nn.ReLU(),

    nn.Linear(1324, 7),
).to(device)
model.double()

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

val_metrics = {
    "accuracy": Accuracy(),
    "nll": Loss(nn.CrossEntropyLoss())
}

for i in range(1, 8):
    print(f"Top {i} Accuracy: {get_top_n_accuracy(i)}")

for i in range(1, 8):
    print(f"Top {i} Median slowdown: {np.median(get_top_n_slowdown(i))}")

print("Median slowdown: ", np.median(get_slowdown()))

for i in range(1, 8):
    print(f"Top {i} Mean slowdown: {np.mean(get_top_n_slowdown(i))}")

print("Mean slowdown: ", np.mean(get_slowdown()))

