import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def ensure_dir_exists(file_path):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)


def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 20))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    ensure_dir_exists(output_path)
    plt.savefig(output_path)
    plt.close()
    return cm


def plot_average_confusion_matrix(confusion_matrices, class_names, output_path):
    avg_confusion_matrix = np.mean(confusion_matrices, axis=0)
    plt.figure(figsize=(20, 20))
    sns.heatmap(
        avg_confusion_matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Average Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    ensure_dir_exists(output_path)
    plt.savefig(output_path)
    plt.close()


def class_wise_accuracy(y_true, y_pred):
    class_accuracies = {}
    for cls in np.unique(y_true):
        mask = y_true == cls
        class_accuracies[cls] = (y_pred[mask] == y_true[mask]).mean()
    return class_accuracies


def print_class_accuracies(class_accuracies, i2w):
    print("Class-wise accuracies:")
    for cls, acc in class_accuracies.items():
        class_name = i2w.get(cls, f"Unknown-{cls}")
        print(f"Class {cls} ({class_name}): {acc*100:.2f}%")


def analyse_dataset(X, Y):
    unique, counts = np.unique(Y, return_counts=True)
    class_counts = dict(zip(unique, counts))
    print("Samples per class:")
    for cls, count in class_counts.items():
        print(f"Class {cls}: {count} samples")
    print(f"Total samples: {len(Y)}")
    return class_counts
