from data.load_datasets import PreferenceDataset
from models.reward import RewardModel
from tests.test_data import test_preference_dataset_shp
from tests.test_reward_model import test_reward_model_forward

def main():
    # test_preference_dataset_shp()
    test_reward_model_forward()

if __name__ == "__main__":
    main()