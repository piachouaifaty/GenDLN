import os
from pathlib import Path

from dotenv import load_dotenv


root_folder = Path(__file__).resolve().parent.parent.parent.parent
env_file = os.path.join(root_folder, ".env")
load_dotenv(env_file)

qroq_api_key = os.getenv("GROQ_API_KEY")
deepinfra_api_key = os.getenv("DEEPINFRA_API_KEY")
deepinfra_endpoint = os.getenv("DEEPINFRA_ENDPOINT")
azure_openai_base = os.getenv("AZURE_OPENAI_BASE")
azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
azure_openai_version = os.getenv("AZURE_OPENAI_VERSION")
azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

mistral_api_key_v1 = os.getenv("MISTRAL_API_KEY_V1")
mistral_api_key_v2 = os.getenv("MISTRAL_API_KEY_V2")
mistral_api_key_v3 = os.getenv("MISTRAL_API_KEY_V3")
mistral_api_key_v4 = os.getenv("MISTRAL_API_KEY_V4")
mistral_api_key_v5 = os.getenv("MISTRAL_API_KEY_V5")
mistral_api_key_v6 = os.getenv("MISTRAL_API_KEY_V6")
mistral_api_key_v7 = os.getenv("MISTRAL_API_KEY_V7")
mistral_api_key_v8 = os.getenv("MISTRAL_API_KEY_V8")

mistral_api_key_v9 = os.getenv("MISTRAL_API_KEY_V9")

mistral_api_key_v10 = os.getenv("MISTRAL_API_KEY_V10")
mistral_api_key_v11 = os.getenv("MISTRAL_API_KEY_V11")
mistral_api_key_v12 = os.getenv("MISTRAL_API_KEY_V12")
mistral_api_key_v13 = os.getenv("MISTRAL_API_KEY_V13")
mistral_api_key_v14 = os.getenv("MISTRAL_API_KEY_V14")
mistral_api_key_v15 = os.getenv("MISTRAL_API_KEY_V15")
mistral_api_key_v16 = os.getenv("MISTRAL_API_KEY_V16")
mistral_api_key_v17 = os.getenv("MISTRAL_API_KEY_V17")
mistral_api_key_v18 = os.getenv("MISTRAL_API_KEY_V18")
mistral_api_key_v19 = os.getenv("MISTRAL_API_KEY_V19")
mistral_api_key_v20 = os.getenv("MISTRAL_API_KEY_V20")


TRAIN_DATA_PATH = os.path.join(root_folder, "dataset", "claudette_train_merged.tsv")
VAL_DATA_PATH = os.path.join(root_folder,  "dataset", "claudette_val_merged.tsv")
HYPERPARAMETERS_PATH = os.path.join(root_folder, "genetic_dln", "data", "hyperparameters.yaml")
CACHE_FOLDER = os.path.join(root_folder, "genetic_dln", "data", "score_cache")

EVOLUTIONARY_ALGORITHMS_DIR = os.path.join(root_folder, "genetic_dln", "src", "evolutionary_algorithms")
layer1_file_binary = os.path.join(root_folder, "genetic_dln", "data", "base_prompts", "binary", "layer1baseprompts_test.json")
layer2_file_binary = os.path.join(root_folder, "genetic_dln", "data", "base_prompts", "binary", "layer2baseprompts_test.json")

layer1_file_multi_label = os.path.join(root_folder, "genetic_dln", "data", "base_prompts", "multi_label", "layer1baseprompts_test.json")
layer2_file_multi_label = os.path.join(root_folder, "genetic_dln", "data", "base_prompts", "multi_label", "layer2baseprompts_test.json")

balanced_binary_train_set = os.path.join(root_folder, "dataset", "binary_balanced_train_100.json")
multi_label_train_set = os.path.join(root_folder, "dataset", "multi_label_train_100.json")

validation_set_binary = os.path.join(root_folder, "dataset", "binary_balanced_validation_1000.json")
validation_set_multi_label = os.path.join(root_folder, "dataset", "multi_label_validation_1000.json")