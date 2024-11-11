from data.load_datasets import PreferenceDataset

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