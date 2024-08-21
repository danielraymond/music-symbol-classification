import fire
import numpy as np

import config.data_config as config
from classifiers.svm import run_svm
from pretrain import pretrain_model
from utils.patches import create_patches, load_img_pages

ds_name = "b-59-850"

hyperparameter_sets = [
    # {"kernel": (64, 64), "stride": (32, 32), "entropy_threshold": 0.7},
    # {"kernel": (64, 64), "stride": (32, 32), "entropy_threshold": 0.8},
    # {"kernel": (64, 64), "stride": (32, 32), "entropy_threshold": 0.9},
    {"kernel": (48, 48), "stride": (24, 24), "entropy_threshold": 0.75},
    {"kernel": (56, 56), "stride": (28, 28), "entropy_threshold": 0.7},
    {"kernel": (56, 56), "stride": (28, 28), "entropy_threshold": 0.75},
    {"kernel": (56, 56), "stride": (28, 28), "entropy_threshold": 0.8},
    {"kernel": (72, 72), "stride": (36, 36), "entropy_threshold": 0.65},
    {"kernel": (72, 72), "stride": (36, 36), "entropy_threshold": 0.7},
    {"kernel": (72, 72), "stride": (36, 36), "entropy_threshold": 0.75},
    {"kernel": (80, 80), "stride": (40, 40), "entropy_threshold": 0.6},
    {"kernel": (80, 80), "stride": (40, 40), "entropy_threshold": 0.65},
    {"kernel": (80, 80), "stride": (40, 40), "entropy_threshold": 0.7},
]


def run_optimisation(epochs: int = 30):
    best_score = 0
    best_hyperparams = None

    for hyperparams in hyperparameter_sets:
        train_crops = extract_crops_with_hyperparameters(hyperparams)

        encoder_filepath = train_self_supervised_cnn(hyperparams, train_crops, epochs)

        classification_results = run_svm(
            ds_name=ds_name,
            samples_per_class=30,
            checkpoint_path=encoder_filepath,
            num_runs=30,
            write_results_to_file=False,
        )
        accuracy = np.mean(classification_results)

        print(f"Hyperparameters: {hyperparams}, Average Score: {accuracy}")

        if accuracy > best_score:
            best_score = accuracy
            best_hyperparams = hyperparams

    print(f"Best Hyperparameters: {best_hyperparams}, Best Score: {best_score}")


def extract_crops_with_hyperparameters(hyperparams):
    kernel = hyperparams["kernel"]
    stride = hyperparams["stride"]
    entropy_threshold = hyperparams["entropy_threshold"]
    print(
        f"Creating patches with kernel: {kernel}, stride: {stride}, entropy_threshold: {entropy_threshold}"
    )
    images = load_img_pages()
    patches = create_patches(
        images=images,
        kernel=kernel,
        stride=stride,
        entropy_threshold=entropy_threshold,
    )

    return patches


def train_self_supervised_cnn(hyperparams, crops, epochs=20):
    kernel = hyperparams["kernel"]
    stride = hyperparams["stride"]
    entropy_threshold = hyperparams["entropy_threshold"]

    output_dir = config.output_dir / "VICReg" / "optimisation"
    output_dir = output_dir / f"CustomCNN"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = "allpatches_"
    model_name += f"k{'x'.join(map(str, kernel))}_"
    model_name += f"s{'x'.join(map(str, stride))}_"
    model_name += f"et{entropy_threshold}_"
    model_name += f"encdim{1600}_"
    model_name += f"expdim{1024}_"
    model_name += f"bs{16}_"
    model_name += f"ep{20}_"
    model_name += f"sw{10.0}_"
    model_name += f"vw{10.0}_"
    model_name += f"cw{1.0}"
    encoder_filepath = output_dir / f"{model_name}_encoder.pt"

    pretrain_model(
        X=crops,
        encoder_filepath=encoder_filepath,
        epochs=epochs,
        sim_loss_weight=10.0,
        var_loss_weight=10.0,
        cov_loss_weight=1.0,
    )

    return encoder_filepath


if __name__ == "__main__":
    fire.Fire(run_optimisation)
