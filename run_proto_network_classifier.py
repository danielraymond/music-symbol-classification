import sys

sys.path.append("./")

from classifiers.proto_network_classifier import run_proto
from config.experiment_config import DS_PRETRAIN_HPARAMS, DS_TEST_HPARAMS

for ds_name, ds_pretrain_config in DS_PRETRAIN_HPARAMS.items():
    model_type = "CustomCNN"

    for k, n in [
        (1, 2),
        (2, 3),
        (6, 4),
    ]:
        for na in [0, 1, 2, 5, 10, 20]:
            run_proto(
                ds_name=ds_name,
                model_type=model_type,
                checkpoint_path=DS_TEST_HPARAMS[ds_name][
                    f"{model_type.lower()}_patches"
                ],
                num_epochs=100,
                learning_rate=0.001,
                k_shot=k,
                n_query=n,
                num_augmentations=na,
                num_bootstrap=5,
            )
