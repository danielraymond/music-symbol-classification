import gc
import os
import random

import fire
import numpy as np
import torch
import torch.optim as optim

import config.data_config as config
from classifiers.models import FinalPrototypicalNetwork, PrototypicalLoss
from network.augmentation import MildAugmentStage
from network.models import CustomCNN
from utils.analysis import (
    analyse_dataset,
    class_wise_accuracy,
    plot_average_confusion_matrix,
    plot_confusion_matrix,
    print_class_accuracies,
)
from utils.loader import load_supervised_data
from utils.train_utils import (
    augment_data_with_original,
    create_index_to_word_mapping,
    get_images_representations,
    proto_network_split,
    write_plot_results_proto,
)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

torch.manual_seed(1)
random.seed(1)
np.random.seed(1)


def train_epoch(
    proto_net, support_x, support_y, query_x, query_y, optimizer, criterion
):
    proto_net.train()
    n_classes = len(torch.unique(support_y))
    k_shot = len(support_x) // n_classes

    optimizer.zero_grad()
    dists = proto_net(support_x, query_x, n_classes, k_shot)
    loss, acc = criterion(dists, query_y, n_classes)
    loss.backward()
    optimizer.step()

    return loss.item(), acc.item()


def evaluate(proto_net, support_x, support_y, eval_x, eval_y):
    proto_net.eval()
    n_classes = len(torch.unique(support_y))
    k_shot = len(support_x) // n_classes

    with torch.no_grad():
        dists = proto_net(support_x, eval_x, n_classes, k_shot)
        _, predictions = torch.max(-dists, 1)
        accuracy = (predictions == eval_y).float().mean().item()

    return accuracy, predictions


def train_and_test_proto(
    *,
    X: np.ndarray,
    Y: np.ndarray,
    i2w: dict,
    model_type: str = "CustomCNN",
    checkpoint_path: str = "",
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    k_shot: int,
    n_query: int,
    num_augmentations: int = 1,
):
    torch.cuda.empty_cache()
    gc.collect()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"USING DEVICE: {device}")

    print(f"Using a VICReg-pretrained {model_type} to obtain images' representations")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder = CustomCNN(encoder_features=checkpoint["encoder_features"])
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder = encoder.to(device)
    encoder.eval()

    support_x, support_y, query_x, query_y, val_x, val_y = proto_network_split(
        X, Y, k_shot, n_query
    )

    print("\nSupport set information:")
    analyse_dataset(support_x, support_y)
    print("\nQuery set information:")
    analyse_dataset(query_x, query_y)
    print("\nValidation set information:")
    analyse_dataset(val_x, val_y)

    augmenter = MildAugmentStage()

    if num_augmentations > 0:
        support_x, support_y = augment_data_with_original(
            support_x, support_y, num_augmentations, augmenter
        )
        query_x, query_y = augment_data_with_original(
            query_x, query_y, num_augmentations, augmenter
        )

    support_x = get_images_representations(
        encoder=encoder, X=torch.from_numpy(support_x), device=device
    )
    query_x = get_images_representations(
        encoder=encoder, X=torch.from_numpy(query_x), device=device
    )
    val_x = get_images_representations(
        encoder=encoder, X=torch.from_numpy(val_x), device=device
    )

    support_x = torch.from_numpy(support_x).float().to(device)
    support_y = torch.from_numpy(support_y).long().to(device)
    query_x = torch.from_numpy(query_x).float().to(device)
    query_y = torch.from_numpy(query_y).long().to(device)
    val_x = torch.from_numpy(val_x).float().to(device)
    val_y = torch.from_numpy(val_y).long().to(device)

    print(f"Support set shape: {support_x.shape}")
    print(f"Query set shape: {query_x.shape}")
    print(f"Validation set shape: {val_x.shape}")

    input_dim = support_x.shape[1]
    proto_net = FinalPrototypicalNetwork(input_dim).to(device)
    criterion = PrototypicalLoss()
    optimizer = optim.Adam(proto_net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    for epoch in range(num_epochs):
        loss, acc = train_epoch(
            proto_net, support_x, support_y, query_x, query_y, optimizer, criterion
        )
        val_acc, _ = evaluate(proto_net, support_x, support_y, val_x, val_y)
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Train Acc: {acc:.4f}, Val Acc: {val_acc:.4f}"
        )

    train_acc, train_predictions = evaluate(
        proto_net, support_x, support_y, query_x, query_y
    )
    val_acc, val_predictions = evaluate(proto_net, support_x, support_y, val_x, val_y)

    print(f"Final Train Accuracy: {train_acc*100:.2f}%")
    print(f"Final Validation Accuracy: {val_acc*100:.2f}%")

    train_class_acc = class_wise_accuracy(
        query_y.cpu().numpy(), train_predictions.cpu().numpy()
    )
    val_class_acc = class_wise_accuracy(
        val_y.cpu().numpy(), val_predictions.cpu().numpy()
    )

    print("\nTrain set class-wise accuracies:")
    print_class_accuracies(train_class_acc, i2w)
    print("\nValidation set class-wise accuracies:")
    print_class_accuracies(val_class_acc, i2w)

    return train_acc * 100, val_acc * 100, X, val_y, val_predictions


def run_proto(
    *,
    ds_name: str,
    min_occurence: int = 50,
    model_type: str = "CustomCNN",
    checkpoint_path: str = "data/b-59-850/experiments/VICReg/CustomCNN/allpatches_k64x64_s32x32_et0.8_encdim1600_expdim1024_bs256_ep150_sw10_vw10_cw1_encoder_final.pt",
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    k_shot: int = 5,
    n_query: int = 15,
    num_augmentations: int = 1,
    num_bootstrap: int = 5,
):
    torch.cuda.empty_cache()
    gc.collect()

    print("--------PROTO NETWORK CLASSIFICATION EXPERIMENT--------")
    print(f"Dataset: {ds_name}")
    print(f"K-Shot: {k_shot}")
    print(f"N-Query: {n_query}")
    print(f"Num Augmentations: {num_augmentations}")
    print(f"Min occurrence: {min_occurence}")
    print(f"Model type: {model_type}")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Number of Epochs: {num_epochs}")
    print(f"Learning Rate: {learning_rate}")
    print("----------------------------------------------------")

    # 1) LOAD DATA
    data_dict = load_supervised_data(ds_name=ds_name, min_occurence=min_occurence)
    X, Y, w2i = data_dict["X"], data_dict["Y"], data_dict["w2i"]

    i2w = create_index_to_word_mapping(w2i)
    class_names = [i2w[i] for i in range(len(i2w))]

    print(f"Dataset {ds_name} information:")
    print(f"\tTotal number of samples: {len(Y)}")
    print(f"\tNumber of classes: {len(w2i)}")
    print("----------------------------------------------------")

    # 2) SET OUTPUT DIR
    output_dir = config.output_dir / "proto_net"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3) RUN BOOTSTRAP ITERATIONS
    confusion_matrices = []
    train_accuracies = []
    val_accuracies = []

    for i in range(num_bootstrap):
        print(f"\nBootstrap iteration {i+1}/{num_bootstrap}")

        indices = np.random.choice(len(X), len(X), replace=True)
        X_sampled, Y_sampled = X[indices], Y[indices]

        train_accuracy, val_accuracy, _, val_y, val_predictions = train_and_test_proto(
            X=X_sampled,
            Y=Y_sampled,
            i2w=i2w,
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            k_shot=k_shot,
            n_query=n_query,
            num_augmentations=num_augmentations,
        )

        cm = plot_confusion_matrix(
            val_y.cpu().numpy(),
            val_predictions.cpu().numpy(),
            class_names,
            output_dir
            / "confusion_matrices"
            / f"k-{k_shot}_n-{n_query}_num-aug-{num_augmentations}_confusion_matrix{i}.png",
        )
        confusion_matrices.append(cm)

        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

    # 4) CALCULATE STATISTICS
    train_mean = np.mean(train_accuracies)
    train_std = np.std(train_accuracies)
    val_mean = np.mean(val_accuracies)
    val_std = np.std(val_accuracies)

    plot_average_confusion_matrix(
        confusion_matrices,
        class_names,
        output_dir
        / "confusion_matrices"
        / f"k-{k_shot}_n-{n_query}_num-aug-{num_augmentations}_average_confusion_matrix.png",
    )

    # 5) SAVE RESULTS
    print("----------------------------------------------------")
    print("RESULTS SUMMARY:")
    print(f"\tK-Shot: {k_shot}")
    print(f"\tN-Query: {n_query}")
    print(f"\tNum Augmentations: {num_augmentations}")
    print(f"\tMean Train accuracy: {train_mean:.2f}%")
    print(f"\tTrain standard deviation: {train_std:.2f}%")
    print(f"\tMean Validation accuracy: {val_mean:.2f}%")
    print(f"\tValidation standard deviation: {val_std:.2f}%")

    write_plot_results_proto(
        filepath=output_dir / "results.txt",
        from_weights=checkpoint_path,
        k_shot=k_shot,
        n_query=n_query,
        num_augmentations=num_augmentations,
        train_results=train_accuracies,
        val_results=val_accuracies,
    )


if __name__ == "__main__":
    fire.Fire(run_proto)
