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
# muon code

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T

    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(0) > G.size(1):
        X = X.T
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven"t tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                assert g.ndim == 2

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - momentum)
                g = g.lerp(buf, momentum) if nesterov else buf

                g = zeropower_via_newtonschulz5(g, steps=ns_steps)
                p.add_(g, alpha=-lr * max(1, p.size(0) / p.size(1)) ** 0.5)

# -----------------------------------------------------------------------------
# model code

class CastedRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), self.weight.to(dtype=x.dtype), self.eps)

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim))
            t = torch.arange(seq_len, device=x.device).type_as(inv_freq)
            freqs = torch.outer(t, inv_freq)
            self.cos_cached = freqs.cos().to(dtype=x.dtype)
            self.sin_cached = freqs.sin().to(dtype=x.dtype)
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

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
        self.rotary = Rotary(self.head_dim)
        self.q_norm = CastedRMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = CastedRMSNorm(self.head_dim, eps=1e-6)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        cos, sin = self.rotary(q)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config.ffn_dim

        self.w1 = nn.Linear(config.n_embd, 2 * config.ffn_dim, bias=False)
        self.w2 = nn.Linear(config.ffn_dim, config.n_embd, bias=False)

    def forward(self, x):
        x, gate = self.w1(x).chunk(2, dim = -1)
        return self.w2(F.silu(x) * gate)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = CastedRMSNorm(config.n_embd, eps=1e-6)
        self.attn = CausalSelfAttention(config)
        self.norm2 = CastedRMSNorm(config.n_embd, eps=1e-6)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

@dataclass
class GPTConfig:  # gpt-2 config, about 124m params
    vocab_size : int = 50304
    n_layer : int = 12
    n_head : int = 12
    n_embd : int = 768
    ffn_dim: int = 2048

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            norm = CastedRMSNorm(config.n_embd, eps=1e-6)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.norm(x)

        logits = self.lm_head(x)

        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            return None, loss

        return logits.float(), None

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

    def state_dict(self):
        return dict(current_shard=self.current_shard, current_position=self.current_position)

    def load_state_dict(self, state):
        self.current_shard = state["current_shard"]
        self.current_position = state["current_position"]
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
    input_bin : str = 'data/fineweb_mix110B/mix_train_*.bin' # input .bin to train on

    # model hyperparams
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 672
    ffn_dim: int = 1840

    # optimization hyperparams
    batch_size : int = 128 # batch size, in sequences, across all devices
    device_batch_size : int = 4 # batch size, in sequences, per device
    sequence_length : int = 1024 # sequence length, in tokens

    num_iterations : int = 80000 # number of iterations to run
    embed_learning_rate : float = 0.002
    scalar_learning_rate : float = 0.002
    muon_learning_rate : float = 0.02
    warmup_iters : int = 1000
    warmdown_iters : int = 10000 # number of iterations of linear warmup/warmdown for triangular or trapezoidal schedule
    weight_decay : float = 0

    checkpoint_path: str = "checkpoint.pt"
    checkpoint_interval : int = 10000 # 0 disables periodic checkpointing
    resume_checkpoint_path: str = ""

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
    resume_checkpoint = torch.load(args.resume_checkpoint_path, map_location="cpu", weights_only=False) if args.resume_checkpoint_path else None
    if resume_checkpoint and "loader_state_dict" in resume_checkpoint:
        train_loader.load_state_dict(resume_checkpoint["loader_state_dict"])
        x, y = resume_checkpoint["x"].cuda(), resume_checkpoint["y"].cuda()
    else:
        x, y = train_loader.next_batch()

    # init the model from scratch
    assert args.n_layer >= 1
    assert args.n_head >= 1
    assert args.n_embd % args.n_head == 0
    assert (args.n_embd // args.n_head) % 2 == 0

    model_config = GPTConfig(**resume_checkpoint["config"]) if resume_checkpoint else GPTConfig(
        vocab_size=50257, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, ffn_dim=args.ffn_dim)

    model = GPT(model_config)
    model = model.cuda()
    raw_model = model
    if resume_checkpoint:
        raw_model.load_state_dict(resume_checkpoint["model_state_dict"])

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:>12,}")
    print(f"Trainable parameters: {trainable_params:>12,}")
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

    # collect the parameters to optimize
    hidden_matrix_params = [p for p in raw_model.transformer.h.parameters() if p.ndim == 2]
    embed_head_params = [raw_model.lm_head.weight]
    scalar_params = [p for p in raw_model.parameters() if p.ndim < 2]

    # init the optimizer(s)
    adam_params = [
        dict(params=embed_head_params, lr=args.embed_learning_rate),
        dict(params=scalar_params, lr=args.scalar_learning_rate),
    ]
    optimizer1 = torch.optim.AdamW(adam_params, betas=(0.8, 0.95), weight_decay=args.weight_decay, fused=True, eps=1e-10)
    optimizer2 = Muon(hidden_matrix_params, lr=args.muon_learning_rate, momentum=0.95)
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
    start_step = 0
    if resume_checkpoint:
        for opt, state in zip(optimizers, resume_checkpoint.get("optimizer_state_dicts", [])):
            opt.load_state_dict(state)
        for sched, state in zip(schedulers, resume_checkpoint.get("scheduler_state_dicts", [])):
            sched.load_state_dict(state)
        start_step = resume_checkpoint.get("step", 0)
        print(f"resumed checkpoint from step {start_step}")

    def save_checkpoint(step):
        base_path = Path(args.checkpoint_path)
        checkpoint_path = base_path.with_name(f"{base_path.stem}_step{step:06d}{base_path.suffix}")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "config": asdict(raw_model.config),
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dicts": [opt.state_dict() for opt in optimizers],
            "scheduler_state_dicts": [sched.state_dict() for sched in schedulers],
            "loader_state_dict": train_loader.state_dict(),
            "x": x.cpu(),
            "y": y.cpu(),
            "step": step,
        }, checkpoint_path)
        print(f"saved checkpoint to {checkpoint_path}")

    training_time_ms = 0
    # start the clock
    torch.cuda.synchronize()
    t0 = time.time()
    # begin training
    if resume_checkpoint is None:
        train_loader.reset()
    for step in range(start_step, args.num_iterations + 1):
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
            save_checkpoint(step)
            break

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        loss_accum = 0.0
        for i in range(train_accumulation_steps):
            # forward pass
            with ctx:
                _, loss = model(x, y)
                loss_accum += loss.detach()
                # advance the dataset for the next batch
            # backward pass
            (loss / train_accumulation_steps).backward()
            x, y = train_loader.next_batch()

        train_loss = loss_accum / train_accumulation_steps

        # step the optimizers and schedulers
        for opt, sched in zip(optimizers, schedulers):
            opt.step()
            sched.step()
        # null the gradients
        model.zero_grad(set_to_none=True)
        # --------------- TRAINING SECTION END -------------------
        if args.checkpoint_interval > 0 and (step + 1) % args.checkpoint_interval == 0:
            save_checkpoint(step + 1)

        approx_time = training_time_ms + 1000 * (time.time() - t0)
        print(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")
