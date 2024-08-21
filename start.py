import sys

sys.path.append("./")

from classifiers.knn import run_knn
from classifiers.svm import run_svm
from classifiers.neural_network_classifier import run_nn
from classifiers.proto_network_classifier import run_proto
from config.experiment_config import DS_PRETRAIN_HPARAMS, DS_TEST_HPARAMS
from pretrain import run_pretrain as pretrain


def run_experiment(ds_pretrain_hparams: dict, ds_test_hparams: dict):
    model_type = "CustomCNN"

    for ds_name, ds_pretrain_config in ds_pretrain_hparams.items():
        pretrain(
            ds_name=ds_name,
            model_type=model_type,
            **ds_pretrain_config,
        )
        run_knn(
            ds_name=ds_name,
            samples_per_class=5,
            model_type=model_type,
            checkpoint_path=ds_test_hparams[ds_name][
                f"{model_type.lower()}_patches"
            ],
            num_runs=5,
            write_results_to_file=True,
            num_augmentations=0
        )
        run_svm(
            ds_name=ds_name,
            samples_per_class=5,
            model_type=model_type,
            checkpoint_path=ds_test_hparams[ds_name][
                f"{model_type.lower()}_patches"
            ],
            num_runs=5,
            write_results_to_file=True,
            num_augmentations=0
        )
        run_nn(
            ds_name=ds_name,
            samples_per_class=5,
            model_type=model_type,
            checkpoint_path=ds_test_hparams[ds_name][
                f"{model_type.lower()}_patches"
            ],
            hidden_dim=1024,
            num_epochs=100,
            learning_rate=0.001,
            num_bootstrap_runs=5,
            num_augmentations=0,
        )
        run_proto(
            ds_name=ds_name,
            model_type=model_type,
            checkpoint_path=ds_test_hparams[ds_name][
                f"{model_type.lower()}_patches"
            ],
            num_epochs=100,
            learning_rate=0.001,
            k_shot=2,
            n_query=3,
            num_augmentations=0,
            num_bootstrap=5,
        )


if __name__ == "__main__":
    run_experiment(DS_PRETRAIN_HPARAMS, DS_TEST_HPARAMS)
