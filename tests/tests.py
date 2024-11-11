
import torch
from models.reward import RewardModel
from data.load_datasets import PreferenceDataset
from models.policy import PolicyModel

def test_preference_dataset_shp():
    dataset = PreferenceDataset(dataset_name='shp', tokenizer_name='gpt2')
    assert len(dataset) > 0, "Dataset should not be empty."
    sample = dataset[0]
    assert 'input_ids' in sample, "'input_ids' key missing in sample."
    assert 'attention_mask' in sample, "'attention_mask' key missing in sample."
    assert 'labels' in sample, "'labels' key missing in sample."
    assert 'preference' in sample, "'preference' key missing in sample."
    assert sample['input_ids'].dim() == 1, "input_ids should be a 1D tensor."
    assert sample['labels'].shape[0] == 2, "labels should have two responses."
    print("Test passed: PreferenceDataset loads SHP dataset correctly.")

def test_reward_model_forward():
    # model_name = 'facebook/opt-125m'
    model_name = 'gpt2'
    model = RewardModel(model_name=model_name)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 50))  # [batch_size=2, seq_length=50]
    attention_mask = torch.ones_like(input_ids)
    rewards = model(input_ids=input_ids, attention_mask=attention_mask)
    assert rewards.shape == (2, 1), f"Expected rewards shape (2, 1), got {rewards.shape}"
    print("Test passed: RewardModel forward pass outputs correct shape.")

def test_policy_model_forward():
    model_name = 'gpt2'
    model = PolicyModel(model_name=model_name)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 50))  # [batch_size=2, seq_length=50]
    attention_mask = torch.ones_like(input_ids)
    logits, values = model(input_ids=input_ids, attention_mask=attention_mask)
    assert logits.shape == (2, 50, model.config.vocab_size), f"Expected logits shape (2, 50, vocab_size), got {logits.shape}"
    assert values.shape == (2, 1), f"Expected values shape (2, 1), got {values.shape}"
    print("Test passed: PolicyModel forward pass outputs correct shapes.")

