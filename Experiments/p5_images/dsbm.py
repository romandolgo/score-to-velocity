"""
DSBM (Shi et al., 2023) on 28 x 28 images: --source at t = 0, --target at t = 1.

Either side is an image set from data.py or "noise" for N(0, I), so the script
covers both generation (noise -> mnist) and translation (mnist -> kmnist).

The reference process is dX = sqrt(eps) dW, so the bridge between x0 and x1 is

    x_t = (1 - t) x0 + t x1 + sqrt(eps t (1 - t)) z.

Two UNets predict the far endpoint of the bridge passing through x_t:

    fwd(x_t, t) ~ E[x1 | x_t],   forward drift   (fwd - x_t) / (1 - t)
    bwd(x_t, t) ~ E[x0 | x_t],   backward drift  (bwd - x_t) / t

The MSE on the endpoint is the bridge matching loss weighted by (1 - t)^2
(resp. t^2): the minimizer is the same Markovian projection, but the target
stays bounded as t -> 1.

Stage 0 is bridge matching on the independent coupling; with eps = 0 it is
Rectified Flow. Every IMF iteration then refits bwd on pairs (x0, fwd sample
started at x0) and fwd on pairs (bwd sample started at x1, x1), so the
regression target is always a real sample.

    python data.py
    python dsbm.py --eps 1 --out runs/eps1
    python figures.py runs/eps1

Writes metrics.json, samples.npz and weights.pt into the --out directory.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "p0"))

from models import TimeEmbedding  # noqa: E402


class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, t_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.time = nn.Linear(t_dim, c_out)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Attention(nn.Module):
    """Single-head self-attention over the pixels of the coarsest feature map."""

    def __init__(self, c: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, c)
        self.qkv = nn.Conv2d(c, 3 * c, 1)
        self.proj = nn.Conv2d(c, c, 1)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).unbind(1)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        )
        return x + self.proj(out.transpose(1, 2).reshape(b, c, h, w))


class UNet(nn.Module):
    """Endpoint predictor (x_t, t) -> x_end for 28 x 28 images: 28 -> 14 -> 7.

    The output is x_t plus a correction that is zero at initialization. t may be
    a float, a 0-d tensor or (B,), as for `models.MLP`.
    """

    def __init__(self, ch: int = 32, mult: tuple[int, ...] = (1, 2, 4)):
        super().__init__()
        t_dim = 4 * ch
        self.time = nn.Sequential(
            TimeEmbedding(ch), nn.Linear(ch, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim)
        )
        self.stem = nn.Conv2d(1, ch, 3, padding=1)
        widths = [ch * m for m in mult]

        self.down, self.pool = nn.ModuleList(), nn.ModuleList()
        c = ch
        for w in widths:
            self.down.append(ResBlock(c, w, t_dim))
            c = w
        for w in widths[:-1]:
            self.pool.append(nn.Conv2d(w, w, 3, stride=2, padding=1))

        self.mid1 = ResBlock(c, c, t_dim)
        self.attn = Attention(c)
        self.mid2 = ResBlock(c, c, t_dim)

        self.up, self.unpool = nn.ModuleList(), nn.ModuleList()
        for w in reversed(widths):
            self.up.append(ResBlock(c + w, w, t_dim))
            c = w
        for w in reversed(widths[1:]):
            self.unpool.append(nn.Conv2d(w, w, 3, padding=1))

        self.out = nn.Sequential(
            nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, 1, 3, padding=1)
        )
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x: Tensor, t: Tensor | float) -> Tensor:
        t = torch.as_tensor(t, dtype=torch.float32, device=x.device)
        emb = self.time(t.reshape(-1).expand(x.shape[0]))
        h = self.stem(x)
        skips = []
        for i, block in enumerate(self.down):
            h = block(h, emb)
            skips.append(h)
            if i < len(self.pool):
                h = self.pool[i](h)
        h = self.mid2(self.attn(self.mid1(h, emb)), emb)
        for i, block in enumerate(self.up):
            h = block(torch.cat([h, skips.pop()], dim=1), emb)
            if i < len(self.unpool):
                h = self.unpool[i](F.interpolate(h, scale_factor=2.0, mode="nearest"))
        return x + self.out(h)


class Learner:
    """A network with its optimizer, AMP scaler and EMA copy."""

    def __init__(self, args: argparse.Namespace, device: torch.device):
        self.net = UNet(args.ch).to(device, memory_format=torch.channels_last)
        self.ema = copy.deepcopy(self.net).requires_grad_(False).eval()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr)
        self.amp = device.type == "cuda"
        self.scaler = torch.amp.GradScaler(enabled=self.amp)
        self.decay = args.ema
        self.updates = 0

    def step(self, xt: Tensor, t: Tensor, target: Tensor, lr: float) -> Tensor:
        for group in self.opt.param_groups:
            group["lr"] = lr
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            pred = self.net(xt, t)
        loss = F.mse_loss(pred.float(), target)
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.scaler.step(self.opt)
        self.scaler.update()

        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        with torch.no_grad():
            for p_ema, p in zip(self.ema.parameters(), self.net.parameters()):
                p_ema.lerp_(p, 1 - decay)
        return loss.detach()


def to_uint8(x: Tensor) -> np.ndarray:
    return ((x.clamp(-1, 1) + 1) * 127.5).round().byte().squeeze(-3).cpu().numpy()


def sample_bridge(
    x0: Tensor, x1: Tensor, eps: float, gen: torch.Generator
) -> tuple[Tensor, Tensor]:
    """Random time t of shape (B,) and a point x_t on the bridge from x0 to x1."""
    t = torch.rand(x0.shape[0], generator=gen, device=x0.device)
    z = torch.randn(x0.shape, generator=gen, device=x0.device)
    tt = t[:, None, None, None]
    return t, (1 - tt) * x0 + tt * x1 + (eps * tt * (1 - tt)).sqrt() * z


@torch.no_grad()
def transport(
    net: nn.Module,
    x: Tensor,
    eps: float,
    n_steps: int,
    forward: bool,
    gen: torch.Generator,
    return_traj: bool = False,
) -> Tensor:
    """Run the learned SDE from x across [0, 1]: forward 0 -> 1, else 1 -> 0.

    With s the elapsed time and x_end the predicted endpoint, a step to s' draws
    from the Brownian bridge between the current point and x_end:

        alpha = (s' - s) / (1 - s)
        x' ~ N(x + alpha (x_end - x), eps alpha (1 - s') I)

    This is Euler-Maruyama for drift (x_end - x) / (1 - s) up to O(ds), with
    noise that vanishes at the last step. eps = 0 gives the Euler ODE solver.
    """
    amp = x.device.type == "cuda"
    grid = torch.linspace(0, 1, n_steps + 1, device=x.device)
    traj = [x]
    for s, s_next in zip(grid[:-1], grid[1:]):
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            x_end = net(x, s if forward else 1 - s)
        x_end = x_end.float().clamp(-1, 1)
        alpha = (s_next - s) / (1 - s)
        noise = torch.randn(x.shape, generator=gen, device=x.device)
        x = x + alpha * (x_end - x) + (eps * alpha * (1 - s_next)).sqrt() * noise
        if return_traj:
            traj.append(x)
    return torch.stack(traj) if return_traj else x


def transport_batched(net, x, eps, n_steps, forward, gen, batch: int = 2500) -> Tensor:
    return torch.cat(
        [transport(net, c, eps, n_steps, forward, gen) for c in x.split(batch)]
    )


class Side:
    """One end of the bridge: an image set scaled to [-1, 1], or N(0, I).

    The last n_eval images are held out: `eval` is used for the transport cost,
    `show` (about 20 of them, balanced over classes) for the figures.
    """

    def __init__(
        self,
        name: str,
        device: torch.device,
        n_eval: int,
        seed: int,
        classes: list[int] | None = None,
    ):
        self.name, self.device = name, device
        if name == "noise":
            g = torch.Generator(device).manual_seed(seed)
            self.train = None
            self.eval = torch.randn(n_eval, 1, 28, 28, generator=g, device=device)
            self.show = self.eval[:20]
            return
        path = HERE / "data" / f"{name}.npz"
        if not path.exists():
            raise SystemExit(f"{path} not found, run `python data.py {name}` first")
        with np.load(path) as f:
            x, y = f["x"], f["y"]
        if classes:
            mask = np.isin(y, classes)
            x, y = x[mask], y[mask]
        x = (torch.from_numpy(x)[:, None].float() / 127.5 - 1).to(device)
        y = torch.from_numpy(y.astype(np.int64))[-n_eval:]
        self.train, self.eval = x[:-n_eval], x[-n_eval:]
        per_class = max(1, 20 // len(y.unique()))
        idx = [(y == c).nonzero()[:per_class, 0] for c in y.unique()]
        self.show = self.eval[torch.stack(idx, dim=1).reshape(-1)]

    def draw(self, n: int, gen: torch.Generator) -> Tensor:
        if self.train is None:
            return torch.randn(n, 1, 28, 28, generator=gen, device=self.device)
        idx = torch.randint(len(self.train), (n,), generator=gen, device=self.device)
        return self.train[idx]


def lr_at(step: int, total: int, peak: float, warmup: int = 100) -> float:
    """Linear warmup, then cosine decay from peak to peak / 10."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--source", default="noise", help="noise or a set from data.py")
    p.add_argument("--target", default="mnist")
    p.add_argument("--eps", type=float, default=1.0, help="bridge variance")
    p.add_argument("--steps", type=int, default=2000, help="bridge matching steps")
    p.add_argument("--imf-iters", type=int, default=3)
    p.add_argument("--imf-steps", type=int, default=300, help="per net, per iteration")
    p.add_argument("--cache", type=int, default=10_000, help="pairs per IMF refit")
    p.add_argument("--nfe", type=int, default=20, help="sampling steps")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--imf-lr", type=float, default=2e-4)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--ch", type=int, default=32, help="UNet base width")
    p.add_argument("--letters", default=None, help="emnist_letters subset, e.g. abcde")
    p.add_argument("--n-eval", type=int, default=2000, help="held-out images per side")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=HERE / "runs" / "eps1")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    gen = torch.Generator(device).manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    letters = [ord(c) - ord("a") for c in (args.letters or "").lower()]
    sides = [
        Side(name, device, args.n_eval, args.seed + 1 + i, letters if "emnist" in name else None)
        for i, name in enumerate((args.source, args.target))
    ]
    side0, side1 = sides
    fwd, bwd = Learner(args, device), Learner(args, device)
    n_params = sum(q.numel() for q in fwd.net.parameters())
    print(
        f"{args.source} -> {args.target} on {device}, eps {args.eps}, "
        f"UNet {n_params / 1e6:.2f}M params x 2"
    )

    independent = (side0.eval - side1.eval).square().mean().item()
    metrics = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "params": n_params,
        "cost_independent": independent,
        "stages": [],
    }
    samples: dict[str, list[np.ndarray]] = {
        k: [] for k in ("fwd_traj", "bwd_traj", "fwd_multi", "bwd_multi")
    }
    start = time.time()

    def evaluate(stage: int, loss_f: float, loss_b: float) -> None:
        """Transport cost on held-out inputs and sample dumps for the figures.

        The sampling noise is reseeded identically at every stage, so stages
        differ only through the networks.
        """
        cost = {}
        for key, net, side, is_fwd in (
            ("fwd", fwd.ema, side0, True),
            ("bwd", bwd.ema, side1, False),
        ):
            g = torch.Generator(device).manual_seed(args.seed + 10)
            out = transport_batched(net, side.eval, args.eps, args.nfe, is_fwd, g)
            cost[key] = (out - side.eval).square().mean().item()
            x = side.show
            traj = transport(net, x, args.eps, args.nfe, is_fwd, g, return_traj=True)
            samples[f"{key}_traj"].append(to_uint8(traj))
            multi = [transport(net, x, args.eps, args.nfe, is_fwd, g) for _ in range(8)]
            samples[f"{key}_multi"].append(to_uint8(torch.stack(multi)))
        metrics["stages"].append(
            {
                "stage": stage,
                "cost_fwd": cost["fwd"],
                "cost_bwd": cost["bwd"],
                "loss_fwd": loss_f,
                "loss_bwd": loss_b,
                "seconds": time.time() - start,
            }
        )
        print(
            f"stage {stage}: cost fwd {cost['fwd']:.4f}, bwd {cost['bwd']:.4f} "
            f"(independent {independent:.4f}), {time.time() - start:.0f} s"
        )
        (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        np.savez_compressed(
            args.out / "samples.npz", **{k: np.stack(v) for k, v in samples.items()}
        )

    loss_f = loss_b = torch.zeros((), device=device)
    for step in range(args.steps):
        x0, x1 = side0.draw(args.batch, gen), side1.draw(args.batch, gen)
        t, xt = sample_bridge(x0, x1, args.eps, gen)
        lr = lr_at(step, args.steps, args.lr)
        loss_f = loss_f.lerp(fwd.step(xt, t, x1, lr), 0.02)
        loss_b = loss_b.lerp(bwd.step(xt, t, x0, lr), 0.02)
        if (step + 1) % 250 == 0 or step + 1 == args.steps:
            rate = (step + 1) / (time.time() - start)
            print(
                f"bridge matching {step + 1}/{args.steps}: loss fwd {loss_f.item():.4f}, "
                f"bwd {loss_b.item():.4f}, {rate:.1f} it/s"
            )
    evaluate(0, loss_f.item(), loss_b.item())

    def refit(learner: Learner, x0: Tensor, x1: Tensor, target_is_x1: bool) -> float:
        """Bridge matching on a fixed set of pairs (x0, x1)."""
        running = torch.zeros((), device=device)
        for _ in range(args.imf_steps):
            idx = torch.randint(len(x0), (args.batch,), generator=gen, device=device)
            a, b = x0[idx], x1[idx]
            t, xt = sample_bridge(a, b, args.eps, gen)
            loss = learner.step(xt, t, b if target_is_x1 else a, args.imf_lr)
            running = running.lerp(loss, 0.02)
        return running.item()

    for it in range(1, args.imf_iters + 1):
        x0 = side0.draw(args.cache, gen)
        x1 = transport_batched(fwd.ema, x0, args.eps, args.nfe, True, gen)
        lb = refit(bwd, x0, x1, target_is_x1=False)

        x1 = side1.draw(args.cache, gen)
        x0 = transport_batched(bwd.ema, x1, args.eps, args.nfe, False, gen)
        lf = refit(fwd, x0, x1, target_is_x1=True)
        evaluate(it, lf, lb)

    torch.save(
        {"fwd": fwd.ema.state_dict(), "bwd": bwd.ema.state_dict(), "args": vars(args)},
        args.out / "weights.pt",
    )
    print(f"done in {time.time() - start:.0f} s, results in {args.out}")


if __name__ == "__main__":
    main()
