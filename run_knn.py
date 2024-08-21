import sys

sys.path.append("./")

from classifiers.knn import run_knn
from config.experiment_config import DS_PRETRAIN_HPARAMS

for ds_name, ds_pretrain_config in DS_PRETRAIN_HPARAMS.items():
    model_type = "CustomCNN"

    for spc in [1, 3, 5, 10]:
        for na in [0, 1, 2, 5, 10, 20]:
            run_knn(
                ds_name=ds_name,
                samples_per_class=spc,
                model_type=model_type,
                checkpoint_path=DS_PRETRAIN_HPARAMS[ds_name][
                    f"{model_type.lower()}_patches"
                ],
                num_augmentations=na,
            )
