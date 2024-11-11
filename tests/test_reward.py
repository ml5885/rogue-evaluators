import torch
from models.reward import RewardModel

def test_reward_model_forward():
    model_name = 'facebook/opt-125m'
    model = RewardModel(model_name=model_name)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 50))  # [batch_size=2, seq_length=50]
    attention_mask = torch.ones_like(input_ids)
    rewards = model(input_ids=input_ids, attention_mask=attention_mask)
    assert rewards.shape == (2, 1), f"Expected rewards shape (2, 1), got {rewards.shape}"
    print("Test passed: RewardModel forward pass outputs correct shape.")

if __name__ == "__main__":
    test_reward_model_forward()
