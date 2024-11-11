import os
from data.load_datasets import PreferenceDataset
from models.reward import RewardModel
from tests.tests import *

def load_env(env_file):
    """Loads environment variables from a .env file."""

    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                os.environ[key] = value

load_env('.env')

def main():
    # test_preference_dataset()
    # test_models()
    test_ppo()

if __name__ == "__main__":
    main()