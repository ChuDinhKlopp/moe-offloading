import time
import argparse
import numpy as np
import torch
import torch.nn as nn


class TopKSequenceGRU(nn.Module):
    """
    Input:  x_ids of shape [B, T, K] where each entry is an integer ID in [0, vocab_size).
    Output: logits of shape [B, K, vocab_size] predicting the next K IDs (top-K) for the NEXT step.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int = 6,
        k: int = 3,
        pooling: str = "mean",   # "mean" or "concat"
        dropout: float = 0.0,
    ):
        super().__init__()
        assert pooling in ("mean", "concat")

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.k = k
        self.pooling = pooling

        self.emb = nn.Embedding(vocab_size, embed_dim)

        step_in_dim = embed_dim if pooling == "mean" else (k * embed_dim)

        self.gru = nn.GRU(
            input_size=step_in_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Predict K ids independently (K heads)
        self.heads = nn.ModuleList([nn.Linear(hidden_size, vocab_size) for _ in range(k)])

    def forward(self, x_ids: torch.Tensor) -> torch.Tensor:
        """
        x_ids: [B, T, K] (int64)
        returns logits: [B, K, vocab_size]
        """
        B, T, K = x_ids.shape
        assert K == self.k

        # Embed: [B, T, K, E]
        x = self.emb(x_ids)

        # Pool per timestep to a single vector:
        if self.pooling == "mean":
            # [B, T, E]
            x_step = x.mean(dim=2)
        else:
            # concat: [B, T, K*E]
            x_step = x.reshape(B, T, K * self.embed_dim)

        # GRU over time: out [B, T, H]
        out, _ = self.gru(x_step)

        # Use last timestep hidden state for next-step prediction: [B, H]
        h_last = out[:, -1, :]

        # K independent heads -> [B, K, vocab]
        logits = torch.stack([head(h_last) for head in self.heads], dim=1)
        return logits


def measure_latency(model, x_ids, warmup=50, iters=200, device="cpu"):
    model.eval()

    with torch.inference_mode():
        # warmup
        for _ in range(warmup):
            _ = model(x_ids)
            if device == "cuda":
                torch.cuda.synchronize()

        # timed runs
        lat_ms = []
        for _ in range(iters):
            if device == "cuda":
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            _ = model(x_ids)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            lat_ms.append((t1 - t0) * 1000.0)

    return np.array(lat_ms)


def main():
    p = argparse.ArgumentParser("Measure latency of 6-layer GRU predicting next top-3 IDs")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq", type=int, default=16, help="sequence length T")
    p.add_argument("--k", type=int, default=3, help="top-k per timestep (default 3)")
    p.add_argument("--vocab", type=int, default=50000, help="number of possible IDs")
    p.add_argument("--embed", type=int, default=128, help="embedding dim")
    p.add_argument("--hidden", type=int, default=256, help="GRU hidden size")
    p.add_argument("--layers", type=int, default=6, help="GRU layers (default 6)")
    p.add_argument("--pooling", type=str, default="mean", choices=["mean", "concat"])
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=200)
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    device = torch.device(args.device)

    model = TopKSequenceGRU(
        vocab_size=args.vocab,
        embed_dim=args.embed,
        hidden_size=args.hidden,
        num_layers=args.layers,
        k=args.k,
        pooling=args.pooling,
        dropout=0.0,
    ).to(device)

    # Example input: [B, T, K] IDs
    # In real usage you’d feed your actual top-3 sequence IDs.
    x_ids = torch.randint(0, args.vocab, (args.batch, args.seq, args.k), dtype=torch.long, device=device)

    lat = measure_latency(model, x_ids, warmup=args.warmup, iters=args.iters, device=args.device)

    print("\n==== Next top-K ID prediction latency ====")
    print(f"Device:  {args.device}")
    print(f"B,T,K:   {args.batch},{args.seq},{args.k}")
    print(f"Vocab:   {args.vocab}")
    print(f"Embed:   {args.embed}")
    print(f"Hidden:  {args.hidden}")
    print(f"Layers:  {args.layers}")
    print(f"Pooling: {args.pooling}")
    print("-----------------------------------------")
    print(f"Mean:    {lat.mean():.3f} ms")
    print(f"Median:  {np.median(lat):.3f} ms")
    print(f"P90:     {np.percentile(lat, 90):.3f} ms")
    print(f"P99:     {np.percentile(lat, 99):.3f} ms")
    print(f"Min/Max: {lat.min():.3f} / {lat.max():.3f} ms")
    print("=========================================\n")

    # Demonstrate getting predicted IDs (top-1 per head) just to show usage:
    with torch.inference_mode():
        logits = model(x_ids)  # [B, K, vocab]
        pred_ids = logits.argmax(dim=-1)  # [B, K]
        print("Example predicted next top-K IDs (argmax per head):")
        print(pred_ids[0].tolist())


if __name__ == "__main__":
    main()

