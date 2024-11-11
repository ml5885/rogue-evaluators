import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import datasets

class PreferenceDataset(Dataset):
    def __init__(self, dataset_name, tokenizer_name, max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.dataset = self.load_dataset(dataset_name)

    def load_dataset(self, dataset_name):
        if dataset_name == 'shp':
            data = datasets.load_dataset('stanfordnlp/SHP', split='train')
        elif dataset_name == 'oasst2':
            data = datasets.load_dataset('OpenAssistant/oasst2', split='train')
        elif dataset_name == 'hh':
            data = datasets.load_dataset('Anthropic/hh-rlhf', split='train')
        else:
            raise ValueError(f"Dataset {dataset_name} not recognized.")
        return data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns:
            dict: {
                'input_ids': Tensor [seq_length],
                'attention_mask': Tensor [seq_length],
                'labels': Tensor [2, seq_length],  # Pairwise responses
                'preference': Tensor [],  # 0 or 1 indicating preference
            }
        """
        item = self.dataset[idx]
        prompt = item.get('prompt', '')
        response_1 = item.get('response_1', '')
        response_2 = item.get('response_2', '')
        preference = item.get('preference', 0)  # Assuming 0 or 1

        # Handle datasets with different field names
        if 'question' in item:
            prompt = item['question']
        if 'best_answer' in item:
            response_1 = item['best_answer']
        if 'wrong_answer' in item:
            response_2 = item['wrong_answer']

        prompt_encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        response_1_encoding = self.tokenizer(
            response_1,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        response_2_encoding = self.tokenizer(
            response_2,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': prompt_encoding['input_ids'].squeeze(0),  # [seq_length]
            'attention_mask': prompt_encoding['attention_mask'].squeeze(0),  # [seq_length]
            'labels': torch.stack([
                response_1_encoding['input_ids'].squeeze(0),  # [seq_length]
                response_2_encoding['input_ids'].squeeze(0)
            ]),  # [2, seq_length]
            'preference': torch.tensor(preference)  # []
        }