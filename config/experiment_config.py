import os

################################################### PRETRAIN HPARAMS:
# Common hyperparameters to all datasets
ENTROPY_THRESHOLD = 0.8
ENCODER_FEATURES_DIM = 1600
EXPANDER_FEATURES_DIM = 1024
EPOCHS = 150
BATCH_SIZE = 256
NUM_RANDOM_PATCHES = -1

# Hyperparameters for each dataset
DS_PRETRAIN_HPARAMS = {
    "b-59-850": {
        "kernel": (64, 64),
        "stride": (32, 32),
        "sim_loss_weight": 10,
        "var_loss_weight": 10,
        "cov_loss_weight": 1,
    },
}


# Add the common hyperparameters to each dataset
for k, v in DS_PRETRAIN_HPARAMS.items():
    v["entropy_threshold"] = ENTROPY_THRESHOLD
    v["encoder_features_dim"] = ENCODER_FEATURES_DIM
    v["expander_features_dim"] = EXPANDER_FEATURES_DIM
    v["epochs"] = EPOCHS
    v["batch_size"] = BATCH_SIZE
    v["num_random_patches"] = NUM_RANDOM_PATCHES


################################################### TEST HPARAMS:

DS_TEST_HPARAMS = {}


def get_model_path(
    *,
    ds_name: str,
    supervised_data: bool = False,
    num_random_patches: int = -1,
    kernel: tuple = (64, 64),
    stride: tuple = (32, 32),
    entropy_threshold: float = 0.8,
    model_type: str = "CustomCNN",
    encoder_features_dim: int = 1600,
    expander_features_dim: int = 1024,
    epochs: int = 150,
    batch_size: int = 16,
    sim_loss_weight: float = 10.0,
    var_loss_weight: float = 10.0,
    cov_loss_weight: float = 1.0,
):
    output_dir = f"data/{ds_name}/experiments"
    output_dir = os.path.join(output_dir, "VICReg")
    output_dir = os.path.join(output_dir, f"{model_type}")

    model_name = ""
    model_name = f"{num_random_patches}" if num_random_patches > 0 else "all"
    model_name += "patches_"
    model_name += f"k{'x'.join(map(str, kernel))}_"
    model_name += f"s{'x'.join(map(str, stride))}_"
    model_name += f"et{entropy_threshold}_"
    model_name += f"encdim{encoder_features_dim}_"
    model_name += f"expdim{expander_features_dim}_"
    model_name += f"bs{batch_size}_"
    model_name += f"ep{epochs}_"
    model_name += f"sw{sim_loss_weight}_"
    model_name += f"vw{var_loss_weight}_"
    model_name += f"cw{cov_loss_weight}"
    encoder_filepath = os.path.join(output_dir, f"{model_name}_encoder.pt")
    return encoder_filepath


for ds_name, pretrain_config in DS_PRETRAIN_HPARAMS.items():
    DS_TEST_HPARAMS[ds_name] = {}
    for model_type in ["CustomCNN"]:
        DS_TEST_HPARAMS[ds_name][f"{model_type.lower()}_patches"] = get_model_path(
            ds_name=ds_name,
            supervised_data=False,
            model_type=model_type,
            **pretrain_config,
        )
