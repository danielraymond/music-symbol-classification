from typing import Tuple

import numpy as np
import torch


def get_images_representations(*, encoder, X, device):
    Y = []
    with torch.no_grad():
        for x in X:
            x = x.unsqueeze(0).to(device)
            y = encoder(x)[0].cpu().detach().numpy()
            Y.append(y)
    return np.asarray(Y).reshape(X.shape[0], -1)


def write_plot_results(
    filepath,
    from_weights,
    results,
    samples_per_class,
    num_augmentations,
):
    if not filepath.exists():
        with open(filepath, "w") as datfile:
            header = [
                "from_weights",
                "samples_per_class",
                "bootstrap_runs",
                "mean_accuracy",
                "std_accuracy",
                "num_augmentations",
            ]
            datfile.write("\t".join(header) + "\n")
    with open(filepath, "a") as datfile:
        values = [
            from_weights,
            samples_per_class,
            len(results),
            np.mean(results),
            np.std(results),
            num_augmentations,
        ]
        datfile.write("\t".join([str(value) for value in values]) + "\n")


def write_plot_results_proto(
    filepath,
    from_weights,
    k_shot,
    n_query,
    num_augmentations,
    train_results,
    val_results,
):
    if not filepath.exists():
        with open(filepath, "w") as datfile:
            header = [
                "from_weights",
                "k_shot",
                "n_query",
                "bootstrap_runs",
                "num_augmentations",
                "train_mean_accuracy",
                "train_std_accuracy",
                "val_mean_accuracy",
                "val_std_accuracy",
            ]
            datfile.write("\t".join(header) + "\n")

    with open(filepath, "a") as datfile:
        train_mean = np.mean(train_results)
        train_std = np.std(train_results)
        val_mean = np.mean(val_results)
        val_std = np.std(val_results)

        values = [
            from_weights,
            k_shot,
            n_query,
            len(val_results),
            num_augmentations,
            f"{train_mean:.2f}",
            f"{train_std:.2f}",
            f"{val_mean:.2f}",
            f"{val_std:.2f}",
        ]
        datfile.write("\t".join([str(value) for value in values]) + "\n")


def train_test_split_balanced(
    X: np.ndarray, Y: np.ndarray, samples_per_class: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    XTrain, YTrain = [], []
    XTest, YTest = [], []
    min_test_samples = float("inf")

    unique_ys = np.unique(Y, axis=0)

    for unique_y in unique_ys:
        indices = np.argwhere(Y == unique_y).flatten()
        np.random.shuffle(indices)

        train_indices = indices[:samples_per_class]
        remaining_indices = indices[samples_per_class:]

        XTrain.extend(X[train_indices])
        YTrain.extend(Y[train_indices])

        min_test_samples = min(min_test_samples, len(remaining_indices))

    for unique_y in unique_ys:
        indices = np.argwhere(Y == unique_y).flatten()
        remaining_indices = indices[samples_per_class:]

        test_indices = remaining_indices[:min_test_samples]

        XTest.extend(X[test_indices])
        YTest.extend(Y[test_indices])

    return (
        np.asarray(XTrain, dtype=np.float32),
        np.asarray(YTrain, dtype=np.int64),
        np.asarray(XTest, dtype=np.float32),
        np.asarray(YTest, dtype=np.int64),
    )


def proto_network_split(X, Y, k_shot, n_query):
    classes = np.unique(Y)
    len(classes)

    support_x, support_y = [], []
    query_x, query_y = [], []
    val_x, val_y = [], []

    min_val_samples = float("inf")
    for cls in classes:
        cls_indices = np.where(Y == cls)[0]
        available_val_samples = len(cls_indices) - k_shot - n_query
        min_val_samples = min(min_val_samples, available_val_samples)

    min_val_samples = max(1, min_val_samples)

    for cls in classes:
        cls_indices = np.where(Y == cls)[0]
        np.random.shuffle(cls_indices)

        support_indices = cls_indices[:k_shot]
        query_indices = cls_indices[k_shot : k_shot + n_query]
        val_indices = cls_indices[k_shot + n_query : k_shot + n_query + min_val_samples]

        support_x.extend(X[support_indices])
        support_y.extend(Y[support_indices])
        query_x.extend(X[query_indices])
        query_y.extend(Y[query_indices])
        val_x.extend(X[val_indices])
        val_y.extend(Y[val_indices])

    return (
        np.array(support_x),
        np.array(support_y),
        np.array(query_x),
        np.array(query_y),
        np.array(val_x),
        np.array(val_y),
    )


def augment_data_with_original(data, labels, num_augmentations, augmenter):
    augmented_data = []
    augmented_labels = []
    for i in range(len(data)):
        original_sample = torch.from_numpy(data[i]).unsqueeze(0).float()

        augmented_data.append(original_sample.numpy())
        augmented_labels.append(labels[i])

        if num_augmentations > 0:
            augmented_samples = augmenter(
                original_sample.repeat(num_augmentations, 1, 1, 1)
            )
            augmented_data.append(augmented_samples.numpy())
            augmented_labels.extend([labels[i]] * num_augmentations)

    return np.concatenate(augmented_data), np.array(augmented_labels)


def transform_data(data, augmenter):
    transformed_data = []
    for sample in data:
        sample_tensor = torch.from_numpy(sample).unsqueeze(0).float()
        transformed_sample = augmenter(sample_tensor)
        transformed_data.append(transformed_sample.squeeze(0).numpy())
    return np.array(transformed_data)


def create_index_to_word_mapping(w2i):
    return {v: k for k, v in w2i.items()}
