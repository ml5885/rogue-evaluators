import torch
from torch.utils.data import DataLoader
from training.logger import setup_logger

logger = setup_logger('evaluation.log')

def evaluate_reward_model(reward_model, test_dataloader, tokenizer, device):
    """
    Evaluates the reward model on the test dataset.
    Computes an average reward score for each preferred response.

    Args:
        reward_model (torch.nn.Module): The trained reward model.
        test_dataloader (DataLoader): DataLoader for the test set.
        tokenizer (AutoTokenizer): Tokenizer used for encoding inputs.
        device (torch.device): Device on which to run evaluation.

    Returns:
        float: Average reward score for preferred responses.
    """
    reward_model.eval()
    total_reward = 0
    count = 0

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            preferred_labels = batch['labels'][batch['preference']].to(device)  # Select preferred responses

            # Compute rewards for preferred responses
            rewards = reward_model(preferred_labels, attention_mask)
            total_reward += rewards.sum().item()
            count += rewards.size(0)

    avg_reward = total_reward / count if count > 0 else 0
    return avg_reward

def evaluate_preference_accuracy(policy_model, test_dataloader, tokenizer, device):
    """
    Evaluates the preference accuracy of the policy model by measuring
    how often it selects the preferred response.

    Args:
        policy_model (torch.nn.Module): The trained policy model.
        test_dataloader (DataLoader): DataLoader for the test set.
        tokenizer (AutoTokenizer): Tokenizer used for encoding inputs.
        device (torch.device): Device on which to run evaluation.

    Returns:
        float: Preference accuracy score.
    """
    policy_model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)  # Shape: [batch_size, 2, max_length]

            # Generate responses and calculate similarity to each label
            outputs = policy_model.generate(input_ids, attention_mask=attention_mask, max_length=input_ids.size(1) + 50)
            generated_texts = [tokenizer.decode(g, skip_special_tokens=True) for g in outputs]

            for i, gen_text in enumerate(generated_texts):
                # Tokenize the generated text and both responses
                gen_ids = tokenizer(gen_text, return_tensors='pt', padding=True, truncation=True, max_length=labels.size(-1)).input_ids.to(device)
                label_0_ids = labels[i, 0, :].unsqueeze(0)
                label_1_ids = labels[i, 1, :].unsqueeze(0)

                # Calculate similarity (e.g., using cosine similarity) between generated text and each response
                sim_0 = torch.cosine_similarity(gen_ids.float(), label_0_ids.float(), dim=-1).mean()
                sim_1 = torch.cosine_similarity(gen_ids.float(), label_1_ids.float(), dim=-1).mean()

                # Determine if the preferred response has higher similarity
                if (sim_0 > sim_1 and batch['preference'][i] == 0) or (sim_1 > sim_0 and batch['preference'][i] == 1):
                    correct += 1
                total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy

def evaluate_model(policy_model, reward_model, test_dataloader, tokenizer, device):
    """
    Wrapper function to evaluate both reward model and preference accuracy.

    Args:
        policy_model (torch.nn.Module): The trained policy model.
        reward_model (torch.nn.Module): The trained reward model.
        test_dataloader (DataLoader): DataLoader for the test set.
        tokenizer (AutoTokenizer): Tokenizer used for encoding inputs.
        device (torch.device): Device on which to run evaluation.

    Returns:
        tuple: (average_reward, preference_accuracy)
    """
    print("\nEvaluating Model on Test Set...")
    avg_reward = evaluate_reward_model(reward_model, test_dataloader, tokenizer, device)
    preference_accuracy = evaluate_preference_accuracy(policy_model, test_dataloader, tokenizer, device)
    print(f"Test Set - Reward Evaluation: {avg_reward:.4f}, Preference Accuracy: {preference_accuracy:.4f}")
    return avg_reward, preference_accuracy