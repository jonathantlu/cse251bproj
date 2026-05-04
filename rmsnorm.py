import time
import argparse
import glob
from dataclasses import dataclass, fields, asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# model code

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(norm(x))
        x = x + self.mlp(norm(x))
        return x

@dataclass
class GPTConfig:  # gpt-2 config, about 124m params
    vocab_size : int = 50304
    n_layer : int = 8
    n_head : int = 8
    n_embd : int = 512

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(1024, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device) # shape (t)
        pos_emb = self.transformer.wpe(pos)

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = x + pos_emb

        for block in self.transformer.h:
            x = block(x)
        x = norm(x)

        logits = self.lm_head(x)
        loss = None

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

class Wrapper(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        config = checkpoint["config"]
        self.model = GPT(GPTConfig(**config))
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def forward(self, idx):
        logits, _ = self.model(idx)
        return logits[:, :, :50257]

def load_model(checkpoint_path: str, device: str = "cuda") -> torch.nn.Module:
    """
    Load your trained model from a checkpoint.

    Args:
        checkpoint_path: Path to your checkpoint.pt file
        device: Device string ("cuda" or "cpu")

    Returns:
        A PyTorch nn.Module in eval mode where:
            model(input_ids) -> logits
            - input_ids: LongTensor of shape (batch_size, sequence_length)
            - logits: FloatTensor of shape (batch_size, sequence_length, 50257)
    """

    checkpoint = torch.load(checkpoint_path, map_location = "cpu", weights_only = False)

    model = Wrapper(checkpoint)

    model.to(device)
    model.eval()
    return model

# -----------------------------------------------------------------------------
# data loader code

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()

# -----------------------------------------------------------------------------
# training code

@dataclass
class Hyperparameters:
    # data hyperparams
    input_bin : str = 'data/fineweb10B/fineweb_train_*.bin' # input .bin to train on

    # model hyperparams
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512

    # optimization hyperparams
    batch_size : int = 8*64 # batch size, in sequences, across all devices
    device_batch_size : int = 4 # batch size, in sequences, per device
    sequence_length : int = 1024 # sequence length, in tokens

    num_iterations : int = 4000 # number of iterations to run
    learning_rate : float = 0.0036
    warmup_iters : int = 250
    warmdown_iters : int = 500 # number of iterations of linear warmup/warmdown for triangular or trapezoidal schedule
    weight_decay : float = 0

    checkpoint_path: str = "checkpoint.pt"

def parse_args() -> Hyperparameters:
    parser = argparse.ArgumentParser()

    for field in fields(Hyperparameters):
        parser.add_argument(
            f"--{field.name}",
            type=type(field.default),
            default=field.default,
        )

    return Hyperparameters(**vars(parser.parse_args()))

if __name__ == "__main__":
    args = parse_args()

    assert torch.cuda.is_available()
    device = "cuda:0"
    torch.cuda.set_device(0)

    process_rank = 0
    num_processes = 1

    # convenience variables
    B, T = args.device_batch_size, args.sequence_length
    # calculate the steps of gradient accumulation required to attain the desired global batch size.
    assert args.batch_size % B == 0
    train_accumulation_steps = args.batch_size // B

    # load tokens
    train_loader = DistributedDataLoader(args.input_bin, B, T, process_rank, num_processes)
    print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
    x, y = train_loader.next_batch()

    # init the model from scratch
    assert args.n_layer >= 1
    assert args.n_head >= 1
    assert args.n_embd % args.n_head == 0
    assert (args.n_embd // args.n_head) % 2 == 0

    model_config = GPTConfig(
        vocab_size=50304,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )

    model = GPT(model_config)
    model = model.cuda()
    raw_model = model
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

    # init the optimizer(s)
    param_dict = {pn: p for pn, p in raw_model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': args.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    optimizer = torch.optim.AdamW(optim_groups, lr=args.learning_rate, betas=(0.9, 0.95), fused=True)

    optimizers = [optimizer]
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

        # bit confusing: we want to make sure to eval on 0th iteration
        # but also after the very last iteration. so we loop for step <= num_iterations
        # instead of just < num_iterations (one extra due to <=), only to do
        # the validation/sampling one last time, and then we break right here as we're done.
        if last_step:
            checkpoint_path = Path(args.checkpoint_path)

            if checkpoint_path.parent != Path("."):
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            checkpoint = {
                "config": asdict(raw_model.config),
                "model_state_dict": raw_model.state_dict(),
                "step": step,
            }

            torch.save(checkpoint, checkpoint_path)
            print(f"saved final checkpoint to {checkpoint_path}")

            break

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        for i in range(1, train_accumulation_steps + 1):
            # forward pass
            with ctx:
                _, loss = model(x, y)
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
