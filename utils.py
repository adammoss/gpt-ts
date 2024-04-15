import torch
from torch.nn import functional as F


def generate(model, n_positions, idx, max_new_tokens, static=None, temperature=1.0, top_k=None):
    # idx is (B, T) array of indices in the current context
    model.eval()
    for _ in range(max_new_tokens):
        # crop idx to the last block_size tokens
        idx_cond = idx[:, -n_positions:]
        # get the predictions
        output = model(idx_cond, static=static)
        # focus only on the last time step
        logits = output.logits[:, -1, :] / temperature  # becomes (B, C)
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)  # (B, C)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
    return idx


def randint(low, high=None, size=None, device=None):
    if high is None:
        high = low
        low = 0
    if size is None:
        size = low.shape if isinstance(low, torch.Tensor) else high.shape
    return torch.randint(2 ** 63 - 1, size=size, device=device) % (high - low) + low


def get_last_in_sequence(mask):
    # Get the last index where the mask = 1 (observed)
    B, T = mask.shape
    return T - torch.argmax(torch.flip((mask == 1).long(), [1]), 1) - 1


def get_random_masked_token(mask):
    return torch.squeeze(torch.multinomial(mask.float(), 1))



