
import torch
from models.reward import RewardModel
from models.policy import PolicyModel
from data.load_datasets import PreferenceDataset
from training.ppo import PPO
from data.load_datasets import PreferenceDataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Subset

def test_preference_dataset():
    datasets_to_test = ['summarize_from_feedback', 'hh-rlhf', 'shp']
    for dataset_name in datasets_to_test:
        dataset = PreferenceDataset(dataset_name=dataset_name)
        assert len(dataset) > 0, f"{dataset_name} dataset should not be empty."
        sample = dataset[0]

        assert 'input_ids' in sample, "'input_ids' key missing in sample."
        assert 'attention_mask' in sample, "'attention_mask' key missing in sample."
        assert 'labels' in sample, "'labels' key missing in sample."
        assert 'preference' in sample, "'preference' key missing in sample."
        assert sample['input_ids'].dim() == 1, "input_ids should be a 1D tensor."
        assert sample['labels'].shape[0] == 2, "labels should have two responses."
        assert sample['input_ids'].numel() > 0, "Prompt input_ids should not be empty."

        print(f"Test passed: PreferenceDataset loads {dataset_name} dataset correctly with non-empty prompts.")

def test_models():
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token
    
    # Initialize PolicyModel and RewardModel
    policy_model = PolicyModel.from_pretrained('gpt2')
    reward_model = RewardModel('gpt2')
    
    # Check if models are properly loaded
    assert policy_model is not None, "PolicyModel initialization failed"
    assert reward_model is not None, "RewardModel initialization failed"
    print("Models initialized successfully.")
    
    # Load a small subset of data for testing
    dataset = PreferenceDataset('shp', tokenizer_name='gpt2', max_length=50)
    subset_indices = list(range(4))  # Small subset for testing
    subset_dataset = Subset(dataset, subset_indices)
    dataloader = DataLoader(subset_dataset, batch_size=2)
    
    # Run a single forward pass on the models to ensure they work
    try:
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(policy_model.device) for k, v in batch.items()}
            
            # Test PolicyModel forward pass
            logits, values = policy_model(batch['input_ids'], attention_mask=batch['attention_mask'])
            assert logits is not None, "PolicyModel forward pass failed to produce logits"
            assert values is not None, "PolicyModel forward pass failed to produce values"
            print("PolicyModel forward pass successful.")

            # Test RewardModel forward pass
            rewards = reward_model(batch['input_ids'], attention_mask=batch['attention_mask'])
            assert rewards is not None, "RewardModel forward pass failed to produce rewards"
            print("RewardModel forward pass successful.")
            
        print("All tests passed successfully.")
    
    except Exception as e:
        print(f"Test failed: {e}")

def test_ppo():
    config = {
        'lr': 1e-5,
        'epochs': 1,
        'batch_size': 2,
        'response_max_length': 50,
        'temperature': 1.0,
        'top_p': 0.9,
        'clip_range': 0.2,
        'vf_coef': 0.5,
        'gamma': 0.99,
        'lam': 0.95,
        'max_grad_norm': 0.5,
    }

    tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token

    policy_model = PolicyModel.from_pretrained('gpt2')
    reward_model = RewardModel('gpt2')
    ppo_trainer = PPO(policy_model, reward_model, tokenizer, config)

    # Load a small subset of data for testing
    dataset = PreferenceDataset('shp', tokenizer_name='gpt2', max_length=50)
    subset_indices = list(range(4))  # Small subset for testing
    subset_dataset = Subset(dataset, subset_indices)
    dataloader = DataLoader(subset_dataset, batch_size=config['batch_size'])

    # Run a single training step
    try:
        print("Running PPO training step...")
        ppo_trainer.train(dataloader)
        print("Test passed: PPO train method runs without errors.")
    except Exception as e:
        print(f"Test failed: PPO train method raised an exception: {e}")
