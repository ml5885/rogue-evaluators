import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import datasets

class PreferenceDataset(Dataset):
    def __init__(self, dataset_name, tokenizer_name='gpt2', max_length=512):
        self.dataset_name = dataset_name.lower()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = max_length
        self.dataset = self.load_dataset(self.dataset_name)

    def load_dataset(self, dataset_name):
        if dataset_name == 'summarize_from_feedback':
            data = datasets.load_dataset('openai/summarize_from_feedback', 'comparisons', split='train')
        elif dataset_name == 'hh-rlhf':
            data = datasets.load_dataset('Anthropic/hh-rlhf', split='train')
        elif dataset_name == 'shp':
            data = datasets.load_dataset('stanfordnlp/SHP', split='train')
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        return data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        if self.dataset_name == 'summarize_from_feedback':
            prompt = item.get('info', {}).get('post', '').strip()
            summaries = item.get('summaries', [])
            if len(summaries) >= 2:
                response_1 = summaries[0].get('text', '').strip()
                response_2 = summaries[1].get('text', '').strip()
                preference = 0 if summaries[0].get('choice', '') == 'better' else 1
            else:
                response_1 = response_2 = ''
                preference = 0
        elif self.dataset_name == 'hh-rlhf':
            prompt = item.get('chosen', '').strip()
            response_1 = item.get('chosen', '').strip()
            response_2 = item.get('rejected', '').strip()
            preference = 0  # 'chosen' is preferred
        elif self.dataset_name == 'shp':
            prompt = item.get('history', '').strip()
            response_1 = item.get('human_ref_A', '').strip()
            response_2 = item.get('human_ref_B', '').strip()
            preference = item.get('labels', 0)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        if not prompt:
            raise ValueError(f"Empty prompt encountered at index {idx} for dataset {self.dataset_name}")

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
            padding='max_length',
            return_tensors='pt'
        )

        response_2_encoding = self.tokenizer(
            response_2,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': prompt_encoding['input_ids'].squeeze(0),
            'attention_mask': prompt_encoding['attention_mask'].squeeze(0),
            'labels': torch.stack([
                response_1_encoding['input_ids'].squeeze(0),
                response_2_encoding['input_ids'].squeeze(0)
            ]),
            'preference': torch.tensor(preference, dtype=torch.long)
        }