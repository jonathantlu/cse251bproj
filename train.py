import time
from dataclasses import dataclass
from pathlib import Path

import torch

from dataloader import DistributedDataLoader
from model import GPT, GPTConfig
from muon import Muon

@dataclass
class Hyperparameters:
    # data hyperparams
    input_bin : str = 'data/fineweb10B/fineweb_train_*.bin' # input .bin to train on
    input_val_bin : str = 'data/fineweb10B/fineweb_val_*.bin' # input .bin to eval validation loss on
    # optimization hyperparams
    batch_size : int = 8*64 # batch size, in sequences, across all devices
    device_batch_size : int = 1 # batch size, in sequences, per device
    sequence_length : int = 1024 # sequence length, in tokens

    num_iterations : int = 6200 # number of iterations to run
    learning_rate : float = 0.0036
    warmup_iters : int = 0
    warmdown_iters : int = 1800 # number of iterations of linear warmup/warmdown for triangular or trapezoidal schedule
    weight_decay : float = 0

    # evaluation and logging hyperparams
    val_loss_every : int = 0 # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens : int = 1024 * 64 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    checkpoint_path: str = "checkpoint.pt"
args = Hyperparameters()

assert torch.cuda.is_available()
device = "cuda:0"
torch.cuda.set_device(0)

process_rank = 0
num_processes = 1

# convenience variables
B, T = args.device_batch_size, args.sequence_length
# calculate the number of steps to take in the val loop.
assert args.val_tokens % (B * T) == 0
val_steps = args.val_tokens // (B * T)
# calculate the steps of gradient accumulation required to attain the desired global batch size.
assert args.batch_size % B == 0
train_accumulation_steps = args.batch_size // B

# load tokens
train_loader = DistributedDataLoader(args.input_bin, B, T, process_rank, num_processes)
val_loader = DistributedDataLoader(args.input_val_bin, B, T, process_rank, num_processes)
print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
x, y = train_loader.next_batch()

# init the model from scratch
num_vocab = 50257
model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=12, n_embd=768))
model = model.cuda()
raw_model = model
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

# init the optimizer(s)
optimizer1 = torch.optim.AdamW(raw_model.lm_head.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.weight_decay, fused=True)
optimizer2 = Muon(raw_model.transformer.h.parameters(), lr=0.1*args.learning_rate, momentum=0.95)
optimizers = [optimizer1, optimizer2]
# learning rate decay scheduler (linear warmup and warmdown)
def get_lr(it):
    assert it <= args.num_iterations
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return (it+1) / args.warmup_iters
    # 2) constant lr for a while
    elif it < args.num_iterations - args.warmdown_iters:
        return 1.0
    # 3) linear warmdown
    else:
        decay_ratio = (args.num_iterations - it) / args.warmdown_iters
        return decay_ratio
schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.time()
# begin training
train_loader.reset()
for step in range(args.num_iterations + 1):
    last_step = (step == args.num_iterations)
    # This effectively ignores timing first 10 steps, which are slower for weird reasons.
    # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
    # steps with dummy data first, and then re-initialize the model and reset the loader.
    if step == 10:
        training_time_ms = 0
        t0 = time.time()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

    # once in a while evaluate the validation dataset
    if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # run validation batches
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        for _ in range(val_steps):
            x_val, y_val = val_loader.next_batch()
            with torch.inference_mode(): 
                with ctx:
                    _, loss = model(x_val, y_val, return_logits=False)
                    val_loss += loss

        val_loss /= val_steps
        # log val loss to console and to logfile
        print(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.time()

    # bit confusing: we want to make sure to eval on 0th iteration
    # but also after the very last iteration. so we loop for step <= num_iterations
    # instead of just < num_iterations (one extra due to <=), only to do
    # the validation/sampling one last time, and then we break right here as we're done.
    if last_step:
        checkpoint_path = Path(args.checkpoint_path)

        if checkpoint_path.parent != Path("."):
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "config": raw_model.config,
            "model_state_dict": raw_model.state_dict(),
            "step": step,
            "val_loss": val_loss.item() if "val_loss" in locals() else None,
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"saved final checkpoint to {checkpoint_path}")

        break

    # --------------- TRAINING SECTION BEGIN -----------------
    model.train()
    for i in range(1, train_accumulation_steps + 1):
        # forward pass
        with ctx:
            _, loss = model(x, y, return_logits=False)
            train_loss = loss.detach()
            # advance the dataset for the next batch
        x, y = train_loader.next_batch()
        # backward pass
        loss.backward()

    for p in model.parameters():
        if p.grad is not None:
            p.grad.div_(train_accumulation_steps)
    # step the optimizers and schedulers
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    # --------------- TRAINING SECTION END -------------------

    approx_time = training_time_ms + 1000 * (time.time() - t0)
    print(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")
