import gc
import random

import fire
import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.svm import SVC

import config.data_config as config
from network.augmentation import MildAugmentStage
from network.models import CustomCNN
from utils.analysis import (
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

torch.manual_seed(1)
random.seed(1)
np.random.seed(1)


def train_and_test_svm(
    *,
    XTrain: np.ndarray,
    YTrain: np.ndarray,
    XTest: np.ndarray,
    YTest: np.ndarray,
    model_type: str = "Flatten",
    checkpoint_path: str = "",
    i2w: dict,
    num_augmentations: int = 1,
):
    torch.cuda.empty_cache()
    gc.collect()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"USING DEVICE: {device}")

    augmenter = MildAugmentStage()

    if num_augmentations > 0:
        XTrain, YTrain = augment_data_with_original(
            XTrain, YTrain, num_augmentations, augmenter
        )

    print(f"Using a VICReg-pretrained {model_type} to obtain images' representations")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder = CustomCNN(encoder_features=checkpoint["encoder_features"])
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder = encoder.to(device)
    encoder.eval()
    XTrain = get_images_representations(
        encoder=encoder, X=torch.from_numpy(XTrain), device=device
    )
    XTest = get_images_representations(
        encoder=encoder, X=torch.from_numpy(XTest), device=device
    )

    svmClassifier = SVC(kernel="linear")
    svmClassifier.fit(XTrain, YTrain)
    predictions = svmClassifier.predict(XTest)
    class_rep = classification_report(
        y_true=YTest, y_pred=predictions, output_dict=True
    )

    accuracy = 100 * class_rep["accuracy"]
    print(f"Accuracy: {accuracy:.2f}")

    class_accuracies = class_wise_accuracy(YTest, predictions)
    print("\nTest set class-wise accuracies:")
    print_class_accuracies(class_accuracies, i2w)

    return accuracy, XTrain, predictions, YTest


def run_svm(
    *,
    ds_name: str,
    samples_per_class: int,
    min_occurence: int = 50,
    model_type: str = "CustomCNN",
    checkpoint_path: str = "data/b-59-850/experiments/VICReg/CustomCNN/allpatches_k64x64_s32x32_et0.8_encdim1600_expdim1024_bs256_ep150_sw10_vw10_cw1_encoder.pt",
    num_runs: int = 5,
    write_results_to_file: bool = True,
    num_augmentations: int = 1,
):
    torch.cuda.empty_cache()
    gc.collect()

    print("--------SVM CLASSIFICATION EXPERIMENT--------")
    print(f"Dataset: {ds_name}")
    print(f"Samples per class: {samples_per_class}")
    print(f"Min occurence: {min_occurence}")
    print(f"Model type: {model_type}")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Number of bootstrap runs: {num_runs}")
    print("----------------------------------------------------")

    print(ds_name)
    print("---------------------------------------------")
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
    output_dir = config.output_dir / "svm"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3) RUN BOOTSTRAP
    results = []
    confusion_matrices = []
    for run in range(num_runs):
        print(f"Bootstrap run {run + 1}/{num_runs}")
        XTrain, YTrain, XTest, YTest = train_test_split_balanced(
            X=X,
            Y=Y,
            samples_per_class=samples_per_class,
        )

        print(f"Train set size: {len(YTrain)}, Test set size: {len(YTest)}")
        unique, counts = np.unique(YTest, return_counts=True)
        print("Test set class distribution:")
        for cls, count in zip(unique, counts):
            print(f"  Class {cls}: {count} samples")

        accuracy, _, predictions, y_true = train_and_test_svm(
            XTrain=XTrain,
            YTrain=YTrain,
            XTest=XTest,
            YTest=YTest,
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            i2w=i2w,
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

    if write_results_to_file:
        # 4) SAVE RESULTS
        print("----------------------------------------------------")
        print("BOOTSTRAP SUMMARY:")
        print(f"\tSamples per class: {samples_per_class}")
        print(f"\tNumber of bootstrap runs: {num_runs}")
        print(f"\tMean accuracy: {np.mean(results):.2f}")
        print(f"\tStandard deviation: {np.std(results):.2f}")
        print(f"\tNumber of augmentations: {num_augmentations}")
        write_plot_results(
            filepath=output_dir / "results.txt",
            from_weights=checkpoint_path,
            results=results,
            samples_per_class=samples_per_class,
            num_augmentations=num_augmentations,
        )

    return results


if __name__ == "__main__":
    fire.Fire(run_svm)
