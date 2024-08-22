import sys

sys.path.append("./")

from classifiers.neural_network_classifier import run_nn
from config.experiment_config import DS_PRETRAIN_HPARAMS, DS_TEST_HPARAMS

for ds_name, ds_pretrain_config in DS_PRETRAIN_HPARAMS.items():
    model_type = "CustomCNN"

    for spc in [1, 3, 5, 10, 15, 20, 25, 30]:
        for na in [0, 1, 2, 5, 10, 20, 50, 100]:
            run_nn(
                ds_name=ds_name,
                samples_per_class=spc,
                model_type=model_type,
                checkpoint_path=DS_TEST_HPARAMS[ds_name][
                    f"{model_type.lower()}_patches"
                ],
                hidden_dim=1024,
                num_epochs=100,
                learning_rate=0.001,
                num_bootstrap_runs=5,
                num_augmentations=na,
            )
