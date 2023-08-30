import infopolluter.utils as utils
import os
import json
import numpy as np

TRAIN_SIZE = [np.round(0.1 * i, 2) for i in range(1, 10, 1)]
THRESHOLD = range(10, 90, 10)
ALPHAS = [np.round(0.1 * i + 0.05, 2) for i in range(1, 10, 1)]
default_kfold = 10

DEFAULT_LOCRED = {
    "edgelist_file": "/N/slate/baotruon/infopolluters/data/rt_cc_size322208.txt",
    "score_type": "mistrust",
    "threshold": 60,
    "alpha": 0.85,
    "kfold": 10,
    "train_size": 0.8,
    "random_state": 42,
}

DEFAULT_GRAPHEMB = {
    "edgelist_file": None,
    "score_type": None,
    "undersample": True,
    "threshold": 60,
    "alpha": 0.85,
    "kfold": 10,
    "train_size": 0.8,
    "random_state": 42,
}

DEFAULT_TEXTEMB = {
    "edgelist_file": None,
    "score_type": None,
    "undersample": True,
    "threshold": 60,
    "alpha": 0.85,
    "kfold": 10,
    "train_size": 0.8,
    "random_state": 42,
}


def make_exps(saving_dir):
    all_exps = {}

    # Varying TRAINSIZE AND THRESHOLD
    all_exps["locred"] = {}
    for idx, train_size in enumerate(TRAIN_SIZE):
        for jdx, threshold in enumerate(THRESHOLD):
            cf = {"train_size": train_size, "threshold": threshold}
            config = utils.update_dict(cf, DEFAULT_LOCRED)

            config_name = f"{idx}{jdx}"
            all_exps["locred"][config_name] = config
            fp = os.path.join(saving_dir, "locred", f"{config_name}.json")
            with utils.safe_open(fp, mode="w") as f:
                json.dump(config, f)

    all_exps["alpha"] = {}
    for idx, alpha in enumerate(ALPHAS):
        cf = {"alpha": alpha}
        config = utils.update_dict(cf, DEFAULT_LOCRED)

        config_name = f"{idx}"
        all_exps["alpha"][config_name] = config
        fp = os.path.join(saving_dir, "alpha", f"{config_name}.json")
        with utils.safe_open(fp, mode="w") as f:
            json.dump(config, f)

    all_exps["graphemb"] = {}
    for idx, train_size in enumerate(TRAIN_SIZE):
        for jdx, threshold in enumerate(THRESHOLD):
            cf = {"train_size": train_size, "threshold": threshold}
            config = utils.update_dict(cf, DEFAULT_GRAPHEMB)

            config_name = f"{idx}{jdx}"
            all_exps["graphemb"][config_name] = config
            fp = os.path.join(saving_dir, "graphemb", f"{config_name}.json")
            with utils.safe_open(fp, mode="w") as f:
                json.dump(config, f)

    all_exps["textemb"] = {}  # right now textemb config is exactly as graphemb
    for idx, train_size in enumerate(TRAIN_SIZE):
        for jdx, threshold in enumerate(THRESHOLD):
            cf = {"train_size": train_size, "threshold": threshold}
            config = utils.update_dict(cf, DEFAULT_TEXTEMB)

            config_name = f"{idx}{jdx}"
            all_exps["textemb"][config_name] = config
            fp = os.path.join(saving_dir, "textemb", f"{config_name}.json")
            with utils.safe_open(fp, mode="w") as f:
                json.dump(config, f)

    fp = os.path.join(saving_dir, "all_configs.json")
    json.dump(all_exps, open(fp, "w"))
    print(f"Finish saving config to {fp}")


if __name__ == "__main__":
    # DEBUG
    # ABS_PATH = ''
    # saving_dir = os.path.join(ABS_PATH, "data_hi")

    ABS_PATH = "/N/slate/baotruon/infopolluters"
    # 09162022
    # saving_dir = os.path.join(ABS_PATH, "config")
    # make_exps(saving_dir)

    saving_dir = os.path.join(ABS_PATH, "config_cc")
    make_exps(saving_dir)
