import torch.nn as nn
import os
from transformers import AutoModel

class RewardModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        
        HF_TOKEN = os.environ.get('HF_TOKEN')
        if not HF_TOKEN:
            raise ValueError("HF_TOKEN not found in environment variables")
        
        self.transformer = AutoModel.from_pretrained(model_name, token=HF_TOKEN)
        self.config = self.transformer.config
        self.reward_head = nn.Linear(self.config.hidden_size, 1)

        nn.init.normal_(self.reward_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.reward_head.bias)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: Tensor [batch_size, seq_length]
            attention_mask: Tensor [batch_size, seq_length]
        Returns:
            rewards: Tensor [batch_size, 1]
        """
        outputs = self.transformer(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device)
        )
        last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_length, hidden_size]
        pooled_output = last_hidden_state[:, -1, :]  # [batch_size, hidden_size]
        rewards = self.reward_head(pooled_output)  # [batch_size, 1]
        return rewards