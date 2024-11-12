import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM
from tqdm import tqdm
from training.logger import setup_logger

class PPO:
    def __init__(self, policy_model, reward_model, tokenizer, config):
        self.policy_model = policy_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.config = config
        self.logger = setup_logger()

        self.optimizer = optim.Adam(self.policy_model.parameters(), lr=config.get('lr', 1e-5))
        self.gamma = config.get('gamma', 0.99)
        self.lam = config.get('lam', 0.95)
        self.clip_range = config.get('clip_range', 0.1)
        self.vf_coef = config.get('vf_coef', 0.1)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)

    def train(self, dataloader):
        self.policy_model.train()
        self.reward_model.eval()

        total_loss = 0
        num_batches = 0

        for epoch in range(self.config['epochs']):
            self.logger.info(f"Starting epoch {epoch}...")

            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
                try:
                    batch = {k: v.to(self.policy_model.device) for k, v in batch.items()}

                    generated_sequences, responses = self.generate_responses(batch['input_ids'], batch['attention_mask'])
                    if generated_sequences is None:
                        continue

                    input_ids = generated_sequences
                    attention_mask = torch.ones_like(input_ids)
                    rewards = self.compute_rewards(input_ids, attention_mask)

                    preference_scale = torch.where(batch['preference'] == 1, torch.tensor(1.0), torch.tensor(-0.8))
                    rewards = rewards * preference_scale.to(rewards.device)

                    with torch.no_grad():
                        old_outputs = self.policy_model(input_ids=input_ids, attention_mask=attention_mask)
                        old_logits = old_outputs[0] if isinstance(old_outputs, tuple) else old_outputs.logits

                    accumulated_loss = 0
                    for update_idx in range(self.config.get('n_updates', 4)):
                        outputs = self.policy_model(input_ids=input_ids, attention_mask=attention_mask)
                        logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits

                        log_probs = torch.nn.functional.log_softmax(logits[:, :-1], dim=-1)
                        old_log_probs = torch.nn.functional.log_softmax(old_logits[:, :-1], dim=-1)

                        active_tokens = input_ids[:, 1:]
                        chosen_log_probs = torch.gather(log_probs, dim=-1, index=active_tokens.unsqueeze(-1)).squeeze(-1)
                        old_chosen_log_probs = torch.gather(old_log_probs, dim=-1, index=active_tokens.unsqueeze(-1)).squeeze(-1)

                        advantages = rewards.unsqueeze(1).expand_as(chosen_log_probs)
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                        ratio = (chosen_log_probs - old_chosen_log_probs).exp().clamp(0.0, 5.0)
                        surr1 = ratio * advantages
                        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
                        policy_loss = -torch.min(surr1, surr2).mean()

                        value_pred = logits.mean(dim=-1)
                        value_loss = torch.nn.functional.huber_loss(value_pred, rewards.unsqueeze(1).expand_as(value_pred))

                        loss = (policy_loss + self.vf_coef * value_loss) / self.config.get('n_updates', 4)

                        loss.backward()
                        accumulated_loss += loss.item()

                        if (update_idx + 1) % self.config.get('n_updates', 4) == 0:
                            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.max_grad_norm)
                            self.optimizer.step()
                            self.optimizer.zero_grad()

                        self.logger.info(f"Update {update_idx + 1} - Loss: {loss.item():.4f}")

                    total_loss += accumulated_loss
                    num_batches += 1

                except Exception as e:
                    self.logger.error(f"Error in batch {batch_idx}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue

            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            self.logger.info(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")

        return self.policy_model, self.reward_model