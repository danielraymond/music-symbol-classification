import gc
import os
import random

import fire
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models import CustomCNN
from sklearn.metrics import classification_report

import config.data_config as config
from classifiers.models import FinalNNClassifier
from network.augmentation import MildAugmentStage
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
    train_test_split_balanced,
    write_plot_results,
)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

torch.manual_seed(1)
random.seed(1)
np.random.seed(1)


def train_and_test_nn(
    *,
    XTrain: np.ndarray,
    YTrain: np.ndarray,
    XTest: np.ndarray,
    YTest: np.ndarray,
    i2w: dict,
    model_type: str = "CustomCNN",
    checkpoint_path: str = "",
    hidden_dim: int = 512,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    num_augmentations: int = 2,
):
    torch.cuda.empty_cache()
    gc.collect()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"USING DEVICE: {device}")

    print("\Train set information:")
    analyse_dataset(XTrain, YTrain)
    print("\Test set information:")
    analyse_dataset(XTest, YTest)

    augment = MildAugmentStage()

    # Augment training data
    if num_augmentations > 0:
        XTrain, YTrain = augment_data_with_original(
            XTrain, YTrain, num_augmentations, augment
        )

    print(f"Using a VICReg-pretrained {model_type} to obtain images' representations")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder = CustomCNN(encoder_features=checkpoint["encoder_features"])
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder = encoder.to(device)
    encoder.eval()

    XTrain_representations = get_images_representations(
        encoder=encoder, X=torch.from_numpy(XTrain).float(), device=device
    )
    XTest_representations = get_images_representations(
        encoder=encoder, X=torch.from_numpy(XTest).float(), device=device
    )

    input_dim = XTrain_representations.shape[1]
    output_dim = len(np.unique(YTrain))

    classifier = FinalNNClassifier(input_dim, hidden_dim, output_dim).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    XTrain_tensor = torch.from_numpy(XTrain_representations).float().to(device)
    YTrain_tensor = torch.from_numpy(YTrain).long().to(device)
    XTest_tensor = torch.from_numpy(XTest_representations).float().to(device)
    YTest_tensor = torch.from_numpy(YTest).long().to(device)

    classifier.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = classifier(XTrain_tensor)
        loss = criterion(outputs, YTrain_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    classifier.eval()
    with torch.no_grad():
        test_outputs = classifier(XTest_tensor)
        predictions = torch.argmax(test_outputs, dim=1)

        y_true = YTest_tensor.cpu().numpy()
        y_pred = predictions.cpu().numpy()

        class_rep = classification_report(
            y_true=y_true, y_pred=y_pred, output_dict=True
        )

    accuracy = 100 * class_rep["accuracy"]
    print(f"Accuracy: {accuracy:.2f}")

    class_accuracies = class_wise_accuracy(y_true, y_pred)
    print("\nClass-wise accuracies:")
    print_class_accuracies(class_accuracies, i2w)

    return accuracy, XTrain_representations, YTrain, y_pred, y_true


def run_nn(
    *,
    ds_name: str,
    samples_per_class: int,
    min_occurence: int = 50,
    model_type: str = "CustomCNN",
    checkpoint_path: str = "data/b-59-850/experiments/VICReg/CustomCNN/allpatches_k64x64_s32x32_et0.8_encdim1600_expdim1024_bs256_ep150_sw10_vw10_cw1_encoder_final.pt",
    hidden_dim: int = 512,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    num_bootstrap_runs: int = 10,
    num_augmentations: int = 2,
):
    torch.cuda.empty_cache()
    gc.collect()

    print("--------NN CLASSIFICATION EXPERIMENT--------")
    print(f"Dataset: {ds_name}")
    print(f"Samples per class: {samples_per_class}")
    print(f"Min occurence: {min_occurence}")
    print(f"Model type: {model_type}")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Hidden Dimension: {hidden_dim}")
    print(f"Number of Epochs: {num_epochs}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Number of bootstrap runs: {num_bootstrap_runs}")
    print(f"Number of augmentations: {num_augmentations}")
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
    output_dir = config.output_dir / "nn"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3) RUN BOOTSTRAP
    results = []
    confusion_matrices = []

    for run in range(num_bootstrap_runs):
        print(f"Bootstrap run {run + 1}/{num_bootstrap_runs}")
        XTrain, YTrain, XTest, YTest = train_test_split_balanced(
            X=X,
            Y=Y,
            samples_per_class=samples_per_class,
        )
        (
            accuracy,
            XTrain_embedded,
            augmented_YTrain,
            predictions,
            y_true,
        ) = train_and_test_nn(
            XTrain=XTrain,
            YTrain=YTrain,
            XTest=XTest,
            YTest=YTest,
            i2w=i2w,
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            hidden_dim=hidden_dim,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            num_augmentations=num_augmentations,
        )
        results.append(accuracy)

        cm = plot_confusion_matrix(
            y_true,
            predictions,
            class_names,
            output_dir
            / "confusion_matrices"
            / f"spc-{samples_per_class}_num-aug-{num_augmentations}_confusion_matrix{run}.png",
        )
        confusion_matrices.append(cm)

    plot_average_confusion_matrix(
        confusion_matrices,
        class_names,
        output_dir
        / "confusion_matrices"
        / f"spc-{samples_per_class}_num-aug-{num_augmentations}_average_confusion_matrix.png",
    )

    # 5) SAVE RESULTS
    print("----------------------------------------------------")
    print("BOOTSTRAP SUMMARY:")
    print(f"\tSamples per class: {samples_per_class}")
    print(f"\tNumber of bootstrap runs: {num_bootstrap_runs}")
    print(f"\tMean accuracy: {np.mean(results):.2f}")
    print(f"\tStandard deviation: {np.std(results):.2f}")
    print(f"\tNum Augmentations: {num_augmentations}")

    write_plot_results(
        filepath=output_dir / "results.txt",
        from_weights=checkpoint_path,
        results=results,
        samples_per_class=samples_per_class,
        num_augmentations=num_augmentations,
    )


if __name__ == "__main__":
    fire.Fire(run_nn)
