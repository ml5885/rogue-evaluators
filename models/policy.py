import torch.nn as nn
import torch.nn as nn
import os
from transformers import AutoModelForCausalLM

class PolicyModel(nn.Module):
    def __init__(self, model_name):
        super(PolicyModel, self).__init__()
        
        HF_TOKEN = os.environ.get('HF_TOKEN')
        if not HF_TOKEN:
            raise ValueError("HF_TOKEN not found in .env file")
        
        self.transformer = AutoModelForCausalLM.from_pretrained(model_name, token=HF_TOKEN)
        self.config = self.transformer.config
        self.value_head = nn.Linear(self.config.hidden_size, 1)

        nn.init.normal_(self.value_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, input_ids, attention_mask=None):
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        hidden_states = transformer_outputs.hidden_states[-1]
        value = self.value_head(hidden_states[:, -1, :])
        
        return transformer_outputs.logits, value
