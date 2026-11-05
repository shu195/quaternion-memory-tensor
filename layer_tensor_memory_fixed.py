"""
Investigating the inductive bias fix for the memory tensor:
1. Initialize alpha gate bias to +4.0 so alpha ~ 0.98 (near-lossless retention).
2. Gate write: beta initialized with negative bias (-2.0) so it doesn't blindly write noise during delay steps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

class GatedMemoryLayer(nn.Module):
    def __init__(self, d_in, d_key, d_val, d_out):
        super().__init__()
        self.d_key = d_key
        self.d_val = d_val
        self.W_q = nn.Linear(d_in, d_key, bias=False)
        self.W_k = nn.Linear(d_in, d_key, bias=False)
        self.W_v = nn.Linear(d_in, d_val, bias=False)
        
        # Inductive bias: initialize retention high (alpha ~ 0.98), write conservative (beta ~ 0.1)
        self.gate_alpha = nn.Linear(d_in, 1)
        nn.init.constant_(self.gate_alpha.bias, 4.0)
        
        self.gate_beta = nn.Linear(d_in, 1)
        nn.init.constant_(self.gate_beta.bias, -1.0)
        
        self.out_proj = nn.Linear(d_in + d_val, d_out)
        self.ln = nn.LayerNorm(d_out)

    def forward(self, x_seq):
        B, T, _ = x_seq.shape
        M = torch.zeros(B, self.d_val, self.d_key, device=x_seq.device)
        outputs = []
        for t in range(T):
            xt = x_seq[:, t, :]
            q = F.normalize(self.W_q(xt), dim=-1)
            k = F.normalize(self.W_k(xt), dim=-1)
            v = self.W_v(xt)
            
            alpha = torch.sigmoid(self.gate_alpha(xt)).unsqueeze(-1)
            beta = torch.sigmoid(self.gate_beta(xt)).unsqueeze(-1)
            
            r = torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)
            pred_v = torch.bmm(M, k.unsqueeze(-1)).squeeze(-1)
            err = v - pred_v
            
            M = alpha * M + beta * torch.bmm(err.unsqueeze(-1), k.unsqueeze(1))
            fused = torch.cat([xt, r], dim=-1)
            outputs.append(self.ln(F.gelu(self.out_proj(fused))))
        return torch.stack(outputs, dim=1), M

class GatedModel(nn.Module):
    def __init__(self, d_in, d_model, d_key, d_val):
        super().__init__()
        self.l1 = GatedMemoryLayer(d_in, d_key, d_val, d_model)
        self.head = nn.Linear(d_model, d_val)
    def forward(self, x):
        h, _ = self.l1(x)
        return self.head(h)

def generate_data(batch_size=64, num_kv_pairs=3, delay_steps=15, d_key=16, d_val=16):
    d_in = d_key + d_val + 2
    T = num_kv_pairs + delay_steps + 1
    X = torch.zeros(batch_size, T, d_in)
    Y = torch.zeros(batch_size, d_val)
    for b in range(batch_size):
        keys = F.normalize(torch.randn(num_kv_pairs, d_key), dim=-1)
        values = torch.randn(num_kv_pairs, d_val)
        for i in range(num_kv_pairs):
            X[b, i, :d_key] = keys[i]
            X[b, i, d_key:d_key+d_val] = values[i]
            X[b, i, -2] = 1.0 # Store flag
        for d in range(delay_steps):
            t = num_kv_pairs + d
            X[b, t, -2:] = 0.0 # delay
        q_idx = np.random.randint(0, num_kv_pairs)
        X[b, -1, :d_key] = keys[q_idx]
        X[b, -1, -1] = 1.0 # Query flag
        Y[b] = values[q_idx]
    return X, Y

model = GatedModel(34, 32, 16, 16)
opt = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1, 61):
    X, Y = generate_data(64, num_kv_pairs=3, delay_steps=15, d_key=16, d_val=16)
    opt.zero_grad()
    pred = model(X)[:, -1, :]
    loss = F.mse_loss(pred, Y)
    loss.backward()
    opt.step()
    if epoch % 10 == 0:
        cos = F.cosine_similarity(pred, Y, dim=-1).mean().item()
        print(f"Epoch {epoch:2d} | Loss: {loss.item():.4f} | Cosine Sim: {cos:+.4f}")
