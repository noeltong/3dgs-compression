import argparse
import datetime
import importlib.util
import os
import warnings

from run_lib import eval, train


def _load_config(config_path):
    module_name = "user_config"
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to load config from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "get_config"):
        raise ValueError(f"Config file must define get_config(): {config_path}")
    return module.get_config()


def _parse_args():
    parser = argparse.ArgumentParser(description="3D Gaussian PAM/PAT compression entrypoint.")
    parser.add_argument("--config", required=True, help="Path to a Python config file.")
    parser.add_argument("--workdir", default=None, help="Work directory to store files.")
    parser.add_argument("--mode", choices=["train", "eval"], required=True, help="Run mode.")
    return parser.parse_args()


def main():
    warnings.filterwarnings("ignore")
    args = _parse_args()
    config = _load_config(args.config)

    if args.workdir is not None:
        work_dir = os.path.join("workspace", args.workdir)
    else:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S")
        work_dir = os.path.join("workspace", f"run_{time_str}")

    if args.mode == "train":
        train(config=config, workdir=work_dir)
    else:
        eval(config=config, workdir=work_dir)


if __name__ == "__main__":
    main()
