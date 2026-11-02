"""SEBFormer: source-specific encoders, shared bottleneck fusion and a linear head."""

import hashlib
import math
from typing import Sequence

import torch
from torch import nn


class PrimaryCapsules(nn.Module):
    def __init__(self, in_dim: int, num_caps: int = 4, cap_dim: int = 16):
        super().__init__()
        self.num_caps = num_caps
        self.cap_dim = cap_dim
        self.proj = nn.Linear(in_dim, num_caps * cap_dim)

    def forward(self, x):
        batch = x.size(0)
        out = self.proj(x).view(batch, self.num_caps, self.cap_dim)
        norm = out.norm(dim=-1, keepdim=True) + 1e-8
        return (out / norm) * (norm.square() / (1.0 + norm.square()))


class DigitCapsules(nn.Module):
    def __init__(
        self, in_caps: int, in_dim: int, out_dim: int = 256, routing_iters: int = 3
    ):
        super().__init__()
        self.in_caps = in_caps
        self.routing_iters = routing_iters
        self.W = nn.Parameter(0.02 * torch.randn(1, in_caps, 1, out_dim, in_dim))

    def forward(self, u):
        batch = u.size(0)
        W = self.W.expand(batch, -1, -1, -1, -1)
        u_hat = torch.matmul(W, u.unsqueeze(2).unsqueeze(-1)).squeeze(-1)
        logits = torch.zeros(batch, self.in_caps, 1, device=u.device, dtype=u.dtype)
        for _ in range(self.routing_iters):
            coeff = torch.softmax(logits, dim=2).unsqueeze(-1)
            s = (coeff * u_hat).sum(dim=1)
            norm = s.norm(dim=-1, keepdim=True) + 1e-8
            v = (s / norm) * (norm.square() / (1.0 + norm.square()))
            logits = logits + (u_hat * v.unsqueeze(1)).sum(-1)
        return v.squeeze(1)


class TabularCapsuleExtractor(nn.Module):
    def __init__(
        self,
        dims: Sequence[int],
        out_dim: int = 256,
        num_caps_per_group: int = 4,
        cap_dim: int = 16,
        routing_iters: int = 3,
    ):
        super().__init__()
        self.groups = nn.ModuleList(
            [
                PrimaryCapsules(dim, num_caps=num_caps_per_group, cap_dim=cap_dim)
                for dim in dims
            ]
        )
        self.digit_caps = DigitCapsules(
            in_caps=len(dims) * num_caps_per_group,
            in_dim=cap_dim,
            out_dim=out_dim,
            routing_iters=routing_iters,
        )

    def forward(self, features: Sequence[torch.Tensor]):
        primaries = [group(feature) for group, feature in zip(self.groups, features)]
        return self.digit_caps(torch.cat(primaries, dim=1))


def encoder(dim, heads, depth, dropout):
    layer = nn.TransformerEncoderLayer(
        dim,
        heads,
        dim * 4,
        dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(
        layer, depth, norm=nn.LayerNorm(dim), enable_nested_tensor=False
    )


class EventHierarchy(nn.Module):
    def __init__(self, vocab, dim, heads, dropout, local_layers=2, global_layers=1):
        super().__init__()
        self.dim = dim
        self.embedding = nn.Embedding(vocab, dim, padding_idx=0)
        self.position = nn.Parameter(torch.randn(1, 128, dim) * 0.02)
        self.time = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.local_cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.global_cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.empty = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.local = encoder(dim, heads, local_layers, dropout)
        self.global_encoder = encoder(dim, heads, global_layers, dropout)
        self.norm = nn.LayerNorm(dim)
        self.register_buffer(
            "frequencies",
            torch.exp(
                torch.arange(dim // 2).float()
                * (-math.log(100.0) / max(1, dim // 2 - 1))
            ),
        )

    def forward(self, seq):
        # Token 0 denotes padding; nonzero IDs must be in the fitted vocabulary.
        tokens = seq[..., 0].long()
        valid = tokens != 0
        times = torch.where(valid, seq[..., 1], torch.zeros_like(seq[..., 1]))
        order = times.masked_fill(~valid, -torch.inf).argsort(
            dim=1, descending=True, stable=True
        )
        tokens, times, valid = [a.gather(1, order) for a in (tokens, times, valid)]
        batch, length = tokens.shape
        assert length == 128
        angle = times.unsqueeze(-1).clamp(-100.0, 100.0) * self.frequencies
        x = (
            self.embedding(tokens)
            + self.time(torch.cat((angle.sin(), angle.cos()), -1))
            + self.position
        )
        x = x.masked_fill(~valid.unsqueeze(-1), 0.0)
        # Three full overlapping windows cover all 128 events, with no tail loss.
        starts = (0, 32, 64)
        windows = torch.stack([x[:, s : s + 64] for s in starts], 1)
        masks = torch.stack([valid[:, s : s + 64] for s in starts], 1)
        flat = windows.reshape(batch * 3, 64, self.dim)
        local_input = torch.cat((self.local_cls.expand(batch * 3, -1, -1), flat), 1)
        local_valid = torch.cat(
            (
                torch.ones(batch * 3, 1, dtype=torch.bool, device=x.device),
                masks.reshape(batch * 3, 64),
            ),
            1,
        )
        local = self.local(local_input, src_key_padding_mask=~local_valid).reshape(
            batch, 3, 65, self.dim
        )
        window_valid = masks.any(-1)
        global_input = torch.cat(
            (self.global_cls.expand(batch, -1, -1), local[:, :, 0]), 1
        )
        global_valid = torch.cat(
            (torch.ones(batch, 1, dtype=torch.bool, device=x.device), window_valid), 1
        )
        pooled = self.global_encoder(global_input, src_key_padding_mask=~global_valid)[
            :, 0
        ]
        has_record = valid.any(1)
        pooled = self.norm(torch.where(has_record[:, None], pooled, self.empty))
        # Restore local event states in original sorted positions, averaging overlaps.
        event_states = x.new_zeros(batch, length, self.dim)
        overlap = x.new_zeros(1, length, 1)
        for j, start in enumerate(starts):
            event_states[:, start : start + 64] = (
                event_states[:, start : start + 64] + local[:, j, 1:]
            )
            overlap[:, start : start + 64] += 1
        event_states = (event_states / overlap).masked_fill(~valid.unsqueeze(-1), 0.0)
        states = torch.cat((pooled[:, None], event_states), 1)
        state_mask = torch.cat(
            (torch.ones(batch, 1, dtype=torch.bool, device=x.device), valid), 1
        )
        return pooled, states, state_mask, has_record


class TableHierarchy(nn.Module):
    def __init__(self, dims, dim):
        super().__init__()
        self.capsules = TabularCapsuleExtractor(dims, dim, 4, 16, 3)
        self.group_projection = nn.ModuleList([nn.Linear(64, dim) for _ in dims])
        self.group_type = nn.Parameter(torch.randn(1, len(dims), dim) * 0.02)
        self.norm = nn.LayerNorm(dim)

    def forward(self, arrays):
        primary = [layer(x) for layer, x in zip(self.capsules.groups, arrays)]
        pooled = self.norm(self.capsules.digit_caps(torch.cat(primary, 1)))
        groups = torch.stack(
            [layer(x.flatten(1)) for layer, x in zip(self.group_projection, primary)], 1
        )
        states = torch.cat((pooled[:, None], groups + self.group_type), 1)
        return (
            pooled,
            states,
            torch.ones(states.shape[:2], dtype=torch.bool, device=states.device),
        )


def named_module(seed, name, factory):
    number = int.from_bytes(
        hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "big"
    ) % (2**63 - 1)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(number)
        return factory()


class Block(nn.Module):
    def __init__(self, d, h, p):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.attention = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(p), nn.Linear(2 * d, d)
        )
        self.dropout = nn.Dropout(p)

    def forward(self, x, valid, mask=None):
        y = self.norm1(x)
        x = x + self.dropout(
            self.attention(
                y, y, y, key_padding_mask=~valid, attn_mask=mask, need_weights=False
            )[0]
        )
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x.masked_fill(~valid[..., None], 0.0)


class Fusion(nn.Module):
    """Two synchronous rounds through 32 shared bottleneck states."""

    def __init__(self, seed, dim=256, heads=8, dropout=0.1):
        super().__init__()
        self.retained = list(range(5))
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        str(m): named_module(
                            seed,
                            f"fusion.block{r}.source{m}",
                            lambda: Block(dim, heads, dropout),
                        )
                        for m in self.retained
                    }
                )
                for r in range(2)
            ]
        )
        self.z0 = named_module(
            seed, "fusion.z0", lambda: nn.Parameter(torch.randn(1, 32, dim) * 0.02)
        )

    def forward(self, states, masks):
        h = list(states)
        z = self.z0.expand(h[0].shape[0], -1, -1)
        for blocks in self.blocks:
            updated, temporary = [], []
            # Every source reads the same previous-round shared state.
            for m, x, valid_source in zip(self.retained, h, masks):
                joined = torch.cat([x, z], 1)
                valid = torch.cat(
                    [
                        valid_source,
                        torch.ones(z.shape[:2], dtype=torch.bool, device=x.device),
                    ],
                    1,
                )
                out = blocks[str(m)](joined, valid)
                updated.append(out[:, : x.shape[1]])
                temporary.append(out[:, x.shape[1] :])
            h = updated
            z = torch.stack(temporary).mean(0)
        return h, z.mean(1)


class SEBFormer(nn.Module):
    """Four coded sources plus nine baseline domains, with one linear readout.

    Args:
        group_dims: Encoded widths of the nine baseline domains, in input order.
        num_labels: Number of output labels (1,507 in the UKB task).
        vocab_size: Shared vocabulary size including padding index 0.
        seed: Per-fold initialization seed (43--47 for the primary experiment).

    forward takes nine [batch, domain_width] baseline tensors followed by four
    [batch, 128, 2] event tensors: diagnoses, phenotype fields, procedures, drugs.
    It returns main logits and a dictionary containing five auxiliary logits.
    """

    def __init__(
        self, group_dims, num_labels, vocab_size, seed, dim=256, heads=8, dropout=0.1
    ):
        super().__init__()
        if len(group_dims) != 9 or any(type(x) is not int or x < 1 for x in group_dims):
            raise ValueError("group_dims must contain nine positive integer widths")
        if type(num_labels) is not int or num_labels < 1:
            raise ValueError("num_labels must be a positive integer")
        if type(vocab_size) is not int or vocab_size < 2:
            raise ValueError("vocab_size must include padding and at least one token")
        if dim < 2 or heads < 1 or dim % 2 or dim % heads:
            raise ValueError("dim must be even and divisible by heads")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        self.retained = list(range(5))
        self.events = nn.ModuleDict(
            {
                str(m): named_module(
                    seed,
                    f"events.{m}",
                    lambda: EventHierarchy(vocab_size, dim, heads, dropout, 2, 1),
                )
                for m in range(4)
            }
        )
        self.table = named_module(
            seed, "table", lambda: TableHierarchy(group_dims, dim)
        )
        self.aux = nn.ModuleDict(
            {
                str(m): named_module(
                    seed, f"aux.{m}", lambda: nn.Linear(dim, num_labels)
                )
                for m in self.retained
            }
        )
        self.fusion = Fusion(seed, dim, heads, dropout)
        self.norms = nn.ModuleDict({str(m): nn.LayerNorm(dim) for m in self.retained})
        self.common_norm = nn.LayerNorm(dim)
        self.final = named_module(seed, "final", lambda: nn.Linear(6 * dim, num_labels))

    def encode(self, args):
        pooled, states, masks = [], [], []
        for m in self.retained:
            if m < 4:
                p, s, v, _ = self.events[str(m)](args[9 + m])
            else:
                p, s, v = self.table(args[:9])
            pooled.append(p)
            states.append(s)
            masks.append(v)
        return pooled, states, masks

    def forward(self, *args):
        if len(args) != 13:
            raise ValueError(
                "Expected nine baseline tensors followed by four event tensors"
            )
        pooled, states, masks = self.encode(args)
        h, common = self.fusion(states, masks)
        features = torch.cat(
            [self.norms[str(m)](x[:, 0]) for m, x in zip(self.retained, h)]
            + [self.common_norm(common)],
            -1,
        )
        logits = self.final(features)
        branch = torch.stack(
            [self.aux[str(m)](x) for m, x in zip(self.retained, pooled)], 1
        )
        return logits, dict(branch_logits=branch, retained_sources=self.retained)
