import torch.nn as nn
import os
from transformers import AutoModelForCausalLM

class PolicyModel(AutoModelForCausalLM):
    def __init__(self, model_name):
        super().__init__(AutoModelForCausalLM.from_pretrained(model_name).config)
        
        HF_TOKEN = os.environ.get('HF_TOKEN')
        if not HF_TOKEN:
            raise ValueError("HF_TOKEN not found in environment variables")
        
        self.transformer = AutoModelForCausalLM.from_pretrained(model_name, token=HF_TOKEN)
        self.config = self.transformer.config
        self.value_head = nn.Linear(self.config.hidden_size, 1)

        nn.init.normal_(self.value_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.value_head.bias)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids, attention_mask=None):
        transformer_outputs = self.transformer(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            output_hidden_states=True,
            return_dict=True
        )
        
        hidden_states = transformer_outputs.hidden_states[-1]
        value = self.value_head(hidden_states[:, -1, :])
        
        return transformer_outputs.logits, value