import random
import numpy as np
import torch
import torch.nn.functional as F

def seed_experiment(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_loss_and_accuracy(logits, targets, eq_positions, mask, reduction='mean'):
    """
    Computes the mean negative log-likelihood loss and the accuracy on the right-hand side (RHS)
    of each equation in the mini-batch.

    The equation can be : 
        - "[BOS] [a] [+] [b] [=] [r] [EOS] [PAD] [PAD]", in that case target is "[a] [+] [b] [=] [r] [EOS] [PAD] [PAD]"
        - "[BOS] [a] [+] [b] [+] [c] [=] [r] [EOS]", in that case target is "[a] [+] [b] [+] [c] [=] [r] [EOS]"

    Let :
        - B : batch size
        - S : sequence length
        - V : vocabulary size
    
    Parameters
    ----------
    logits : torch.FloatTensor of shape (B, S, V)
        A tensor containing the logits of the next token for all positions in each sequence of the mini-batch.
    targets : torch.LongTensor of shape (B, S)
        A tensor containing the target next tokens for all positions in each sequence of the mini-batch.
    eq_positions : torch.LongTensor of shape (B,)
        The position of the '=' token in each sequence (each sample has exactly one '=').
    mask : torch.LongTensor of shape (B, S)
        A mask indicating valid tokens (1 if valid, 0 for PAD tokens).
    reduction : str, optional
        Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        - 'none': no reduction will be applied
        - 'mean': average the output of the batch dimension. 
        - 'sum': sum the output of the batch dimension.
        
    Returns
    -------
    loss : torch.Tensor of shape (1,) or (B,) depending on the reduction
        The negative log-likelihood loss computed over the valid (non-PAD) RHS tokens.
    accuracy : torch.Tensor of shape (1,) or (B,) depending on the reduction
        The accuracy over the batch where a sequence is counted as correct only if 
        all valid RHS tokens are predicted correctly.
    """
    B, S, V = logits.size()

    # Compute the probabilities & predictions
    probs = F.log_softmax(logits, dim=2)  # shape (B, S, V)
    predictions = torch.argmax(logits, dim=2) # (B, S)
    # We pick out the log probability corresponding to the correct target token Yij from the entire vocabulary of size V.
    # This is equivalent to the first sum in the loss formula
    targets_probs = probs.gather(2, targets.unsqueeze(2)).squeeze(2) # (B, S)

    # RHS Mask
    positions = torch.arange(S, device=targets.device).unsqueeze(0).expand(B, S) # (B, S)
    mask, eq_positions = mask.to(targets.device), eq_positions.to(targets.device)
    rhs_mask = ((positions > eq_positions.unsqueeze(1)) & (mask == 1)).float() # (B, S)

    # Compute the negative log-likelihood loss for each batch
    loss = -((targets_probs * rhs_mask).sum(dim=1)/rhs_mask.sum(dim=1)) # (B,)

    # Compute the accuracy for each batch
    accuracy = torch.prod(((rhs_mask == 0) | (predictions == targets)).float(), dim=1) # (B,)

    assert reduction in ['none', 'mean', 'sum'], f"reduction must be 'none', 'mean' or 'sum', got {reduction}"
    if reduction != 'none': 
        divider = B if reduction == 'mean' else 1
        loss = loss.sum(dim=0) / divider # (1,)
        accuracy = accuracy.sum(dim=0) / divider # (1,)

    return loss, accuracy