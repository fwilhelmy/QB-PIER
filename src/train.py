import torch

from tqdm import tqdm
import os
from collections import defaultdict
from utils import get_loss_and_accuracy
from logzy import save_model, save_metrics
import numpy as np
import json
from pytorch_lightning.callbacks import ModelCheckpoint

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    acc = 0
    loss = 0
    l2_norm = 0
    n = 0

    for batch in loader:
        batch_x, batch_y, eq_positions, mask = batch # (B, S), (B, S), (B,), (B, S)
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        logits, *_ = model(batch_x) # (B, S, V)
        batch_loss, batch_acc = get_loss_and_accuracy(logits, batch_y, eq_positions, mask)
        n += batch_x.shape[0]
        loss += batch_loss.item() * batch_x.shape[0]
        acc += batch_acc * batch_x.shape[0]

        for p in model.parameters():
            if p.requires_grad:
                l2_norm += p.norm(2).item() ** 2  # Square the norm for each parameter tensor
        l2_norm = l2_norm ** 0.5  # Take the square root to get the overall ℓ₂ norm

    return loss / n, acc / n, l2_norm

def run_evaluation(all_metrics, model, train_loaders, test_loaders, device, step, epoch):
    all_metrics["steps"].append(step)
    all_metrics["epochs"].append(epoch)

    for index_op, op in enumerate(train_loaders):
        loss, acc, l2_norm = evaluate(model, train_loaders[op], device)
        all_metrics['train']['loss'][index_op].append(loss)
        all_metrics['train']['accuracy'][index_op].append(acc.item())
        all_metrics['train']['l2_norm'][index_op].append(l2_norm)

    for index_op, op in enumerate(test_loaders):
        loss, acc, l2_norm = evaluate(model, test_loaders[op], device)
        all_metrics['test']['loss'][index_op].append(loss)
        all_metrics['test']['accuracy'][index_op].append(acc.item())
        all_metrics['test']['l2_norm'][index_op].append(l2_norm)
    
def train(model, args, logdir, optimizer, scheduler, train_loader, eval_train_loaders, eval_test_loaders):   
    # Create checkpoint directory if it doesn't exist.
    os.makedirs(logdir, exist_ok=True)

    # Determine total epochs based on n_steps and the number of batches per epoch.
    n_epochs = (args.n_steps + len(train_loader) - 1) // len(train_loader)
    
    if args.verbose: print(f"Number of training epochs ({n_epochs}) & steps ({n_epochs * len(train_loader)})")

    # Lambda function to initialize the metrics array.
    init_metrics = lambda: {
        'loss': [[] for _ in range(len(args.operation_orders))],
        'accuracy': [[] for _ in range(len(args.operation_orders))],
        'l2_norm': [[] for _ in range(len(args.operation_orders))]
    }

    all_metrics = {'train': init_metrics(), 'test': init_metrics(), 'steps': [], 'epochs': [], 'operation_orders': args.operation_orders}

    run_evaluation(all_metrics, model, eval_train_loaders, eval_test_loaders, args.device, 0, 0)

    current_lr = scheduler.optimizer.param_groups[0]["lr"]
    cur_step = 1
    # Store the last evaluation metrics for each set
    cur_metrics = {m: {k: 0 for k in ['accuracy', 'loss']} for m in ['train', 'test']}
    # Lambda functions to compute the mean of a metric over all operation orders for one evaluation set.
    mean = lambda metrics: np.mean([metrics[op][-1] for op in range(len(args.operation_orders))])

    # Early stopping configuration
    # early_stopping = EarlyStopper(verbose=verbose)

    # Main training loop (per epoch)
    pbar = tqdm(range(1, n_epochs+1), desc="Training", total=n_epochs)
    for epoch in pbar:
        for i, batch in enumerate(train_loader):
            batch_x, batch_y, eq_positions, mask = batch # (B, S), (B, S), (B,), (B, S)
            batch_x, batch_y = batch_x.to(args.device), batch_y.to(args.device)

            optimizer.zero_grad(set_to_none=True)
            model.train()

            logits, *_ = model(batch_x) # (B, S, V)
            loss, _ = get_loss_and_accuracy(logits, batch_y, eq_positions, mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
              
            # Evaluate model on training and test sets periodically.
            eval_start = cur_step <= args.eval_start
            eval_step = cur_step % args.eval_step == 0 and cur_step != args.n_steps
            if eval_start or eval_step:
                run_evaluation(all_metrics, model, eval_train_loaders, eval_test_loaders, args.device, cur_step, epoch)

                cur_metrics = {
                    m: {
                        k: mean(all_metrics[m][k])
                        for k in cur_metrics[m]
                    } for m in cur_metrics
                }

                if not eval_start and args.verbose:
                    to_print = "\n" 
                    to_print += " | ".join(f"Train {k} : {v:.5f}" for k, v in cur_metrics['train'].items())
                    to_print += " || "
                    to_print += " | ".join(f"Test {k} : {v:.5f}" for k, v in cur_metrics['test'].items())
                    to_print += f" || lr = {current_lr:.5f}"
                    print(to_print)

            # Save model statistics & checkpoint periodically.
            if cur_step == 1 or (cur_step % args.save_step == 0 and cur_step != args.n_steps):
                file_name = f"{args.exp_name}_state_step={cur_step}_acc={cur_metrics['test']['accuracy']:.5f}_loss={cur_metrics['test']['loss']:.5f}"
                save_model(os.path.join(logdir, file_name), model, optimizer, verbose=args.verbose)

            # update tqdm progress bar
            pbar.set_postfix({'step': cur_step, 'loss': f"{loss:.5f}", 'lr': f"{current_lr:.3f}"})

            cur_step += 1

        # We update the learning rate scheduler once per epoch.
        scheduler.step()
        current_lr = scheduler.optimizer.param_groups[0]["lr"]

        # if early_stopping.early_stop(evaluate(model, test_loader, device)['loss'], epoch): break

    # Compute the global average step time.
    total_time = pbar.format_dict["elapsed"] # total seconds for all epochs
    all_metrics["total_time"], all_metrics["avg_step_time"] = total_time, total_time / args.n_steps

    # Save final model state after training.
    save_model(f"{logdir}/{args.exp_name}_state", model, optimizer, verbose=args.verbose)
    
    run_evaluation(all_metrics, model, eval_train_loaders, eval_test_loaders, args.device, cur_step, epoch)
    save_metrics(all_metrics, f"{logdir}/{args.exp_name}_metrics.json", verbose=args.verbose)

    return all_metrics