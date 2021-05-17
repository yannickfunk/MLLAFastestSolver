import pandas as pd
from dataset_eval import MatrixEvalDataset
import torch
from torch import nn
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


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


def get_classification_report():
    for features, labels in val_loader:
        features = features.to(device)
        predictions = model(features).cpu().detach().numpy().argmax(axis=1)
        labels = labels.numpy()[:, 0]
        return classification_report(labels, predictions, target_names=SOLVERS)


def get_confusion_matrix():
    for features, labels in val_loader:
        features = features.to(device)
        predictions = model(features).cpu().detach().numpy().argmax(axis=1)
        labels = labels.numpy()[:, 0]
        return confusion_matrix(labels, predictions)


def save_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    import matplotlib.pyplot as plt
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True solver')
    plt.xlabel('Predicted solver')
    plt.savefig("cm.png")


SOLVERS = ["bicg", "bicgstab", "cg", "cgs", "fcg", "gmres", "idr"]
df = pd.read_csv("val_set.csv")
eval_set = MatrixEvalDataset(df)
val_loader = torch.utils.data.DataLoader(eval_set, batch_size=len(eval_set),
                                         shuffle=False, pin_memory=True)
model_path = "models/model_reducedset1_1.36_0.5963.pt"
device = torch.device("cuda:2")

model = nn.Sequential(
    nn.BatchNorm1d(9),

    nn.Linear(9, 447),
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

print(get_classification_report())
save_confusion_matrix(get_confusion_matrix(), SOLVERS, normalize=False)