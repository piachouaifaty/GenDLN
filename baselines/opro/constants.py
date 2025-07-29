import os
from pathlib import Path

from dotenv import load_dotenv

root_folder = Path().resolve().parent.parent
env_file = os.path.join(root_folder, ".env")
load_dotenv(env_file)


MRPC_TRAIN_DATA_PATH = os.path.join(root_folder, "dataset", "mrpc", "mrpc_balanced_100_train.json")
MRPC_VAL_DATA_PATH = os.path.join(root_folder,  "dataset", "mrpc", "mrpc_stratified_1000_val.json")