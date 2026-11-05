"""
Empirical Benchmark Showdown:
Quaternion Associative Memory Tensor (H-DeltaNet) vs.
Standard Causal Transformer (Self-Attention) vs.
Real-Valued DeltaNet (Real Associative Memory)

Benchmark Task: Non-Commutative 3D Rotor Trajectory Forecasting (SO(3) Precession Dynamics).
Evaluates: Parameter Efficiency, Geodesic Angle Error, Inference Scaling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

torch.manual_seed(42)
np.random.seed(42)

# =========================================================================
# 1. QUATERNION MATH UTILITIES
# =========================================================================
def quat_mult(q1, q2):
    r1, i1, j1, k1 = q1.unbind(-1)
    r2, i2, j2, k2 = q2.unbind(-1)
    r = r1*r2 - i1*i2 - j1*j2 - k1*k2
    i = r1*i2 + i1*r2 + j1*k2 - k1*j2
    j = r1*j2 - i1*k2 + j1*r2 + k1*i2
    k = r1*k2 + i1*j2 - j1*i2 + k1*r2
    return torch.stack([r, i, j, k], dim=-1)

def quat_conj(q):
    r, i, j, k = q.unbind(-1)
    return torch.stack([r, -i, -j, -k], dim=-1)

def quat_normalize(q, eps=1e-8):
    return q / torch.sqrt(torch.sum(q**2, dim=-1, keepdim=True) + eps)


# =========================================================================
# 2. MODEL 1: QUATERNION ASSOCIATIVE MEMORY TENSOR (H-DeltaNet)
# =========================================================================
class QuaternionLinear(nn.Module):
    def __init__(self, in_q, out_q, bias=True):
        super().__init__()
        self.in_q = in_q
        self.out_q = out_q
        std = 1.0 / np.sqrt(2.0 * in_q)
        self.A = nn.Parameter(torch.randn(out_q, in_q) * (std / 2.0))
        self.B = nn.Parameter(torch.randn(out_q, in_q) * (std / 2.0))
        self.C = nn.Parameter(torch.randn(out_q, in_q) * (std / 2.0))
        self.D = nn.Parameter(torch.randn(out_q, in_q) * (std / 2.0))
        self.bias = nn.Parameter(torch.zeros(out_q, 4)) if bias else None

    def forward(self, x):
        xr, xi, xj, xk = x.unbind(-1)
        yr = F.linear(xr, self.A) - F.linear(xi, self.B) - F.linear(xj, self.C) - F.linear(xk, self.D)
        yi = F.linear(xr, self.B) + F.linear(xi, self.A) - F.linear(xj, self.D) + F.linear(xk, self.C)
        yj = F.linear(xr, self.C) + F.linear(xi, self.D) + F.linear(xj, self.A) - F.linear(xk, self.B)
        yk = F.linear(xr, self.D) - F.linear(xi, self.C) + F.linear(xj, self.B) + F.linear(xk, self.A)
        out = torch.stack([yr, yi, yj, yk], dim=-1)
        return out if self.bias is None else out + self.bias

class HDeltaNet(nn.Module):
    def __init__(self, in_q=1, d_model_q=4, d_key_q=4, d_val_q=4, out_q=1):
        super().__init__()
        self.d_model_q = d_model_q
        self.d_key_q = d_key_q
        self.d_val_q = d_val_q
        
        self.in_proj = QuaternionLinear(in_q, d_model_q)
        self.W_q = QuaternionLinear(d_model_q, d_key_q, bias=False)
        self.W_k = QuaternionLinear(d_model_q, d_key_q, bias=False)
        self.W_v = QuaternionLinear(d_model_q, d_val_q, bias=False)
        
        self.gate_alpha = nn.Linear(4 * d_model_q, 1)
        nn.init.constant_(self.gate_alpha.bias, 3.5)
        self.gate_beta = nn.Linear(4 * d_model_q, 1)
        nn.init.constant_(self.gate_beta.bias, -0.5)
        
        self.out_proj = QuaternionLinear(d_model_q + d_val_q, d_model_q)
        self.head = QuaternionLinear(d_model_q, out_q)

    def forward(self, x_seq):
        B, T, in_q, _ = x_seq.shape
        M = torch.zeros(B, self.d_val_q, self.d_key_q, 4, device=x_seq.device)
        outputs = []
        
        for t in range(T):
            xt = x_seq[:, t, :, :]
            h_in = F.gelu(self.in_proj(xt)) # (B, d_model_q, 4)
            h_flat = h_in.reshape(B, 4 * self.d_model_q)
            
            q = quat_normalize(self.W_q(h_in))
            k = quat_normalize(self.W_k(h_in))
            v = self.W_v(h_in)
            
            alpha = torch.sigmoid(self.gate_alpha(h_flat)).view(B, 1, 1, 1)
            beta = torch.sigmoid(self.gate_beta(h_flat)).view(B, 1, 1, 1)
            
            # Read
            v_exp_q = q.unsqueeze(1)
            r = torch.sum(quat_mult(M, v_exp_q), dim=2)
            
            # Write (Delta rule)
            v_exp_k = k.unsqueeze(1)
            pred_v = torch.sum(quat_mult(M, v_exp_k), dim=2)
            err = v - pred_v
            
            dM = quat_mult(err.unsqueeze(2), quat_conj(k).unsqueeze(1))
            M = alpha * M + beta * dM
            
            fused = torch.cat([h_in, r], dim=1)
            h_out = torch.tanh(self.out_proj(fused))
            pred_q = self.head(h_out)
            outputs.append(pred_q)
            
        return torch.stack(outputs, dim=1)


# =========================================================================
# 3. MODEL 2: STANDARD CAUSAL TRANSFORMER (Self-Attention)
# =========================================================================
class CausalTransformer(nn.Module):
    def __init__(self, in_dim=4, d_model=16, nhead=4, num_layers=2, out_dim=4):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        # Learnable positional embedding up to length 64
        self.pos_emb = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=0.0, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, x):
        # x: (B, T, 4)
        B, T, _ = x.shape
        h = self.in_proj(x) + self.pos_emb[:, :T, :]
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        out = self.transformer(h, mask=mask, is_causal=True)
        return self.head(out)


# =========================================================================
# 4. MODEL 3: REAL-VALUED DELTANET (Real Associative Memory)
# =========================================================================
class RealDeltaNet(nn.Module):
    def __init__(self, in_dim=4, d_model=16, d_key=16, d_val=16, out_dim=4):
        super().__init__()
        self.d_key = d_key
        self.d_val = d_val
        self.in_proj = nn.Linear(in_dim, d_model)
        self.W_q = nn.Linear(d_model, d_key, bias=False)
        self.W_k = nn.Linear(d_model, d_key, bias=False)
        self.W_v = nn.Linear(d_model, d_val, bias=False)
        
        self.gate_alpha = nn.Linear(d_model, 1)
        nn.init.constant_(self.gate_alpha.bias, 3.5)
        self.gate_beta = nn.Linear(d_model, 1)
        nn.init.constant_(self.gate_beta.bias, -0.5)
        
        self.out_proj = nn.Linear(d_model + d_val, d_model)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, x_seq):
        B, T, _ = x_seq.shape
        M = torch.zeros(B, self.d_val, self.d_key, device=x_seq.device)
        outputs = []
        for t in range(T):
            xt = x_seq[:, t, :]
            h_in = F.gelu(self.in_proj(xt))
            q = F.normalize(self.W_q(h_in), dim=-1)
            k = F.normalize(self.W_k(h_in), dim=-1)
            v = self.W_v(h_in)
            
            alpha = torch.sigmoid(self.gate_alpha(h_in)).unsqueeze(-1)
            beta = torch.sigmoid(self.gate_beta(h_in)).unsqueeze(-1)
            
            r = torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)
            pred_v = torch.bmm(M, k.unsqueeze(-1)).squeeze(-1)
            err = v - pred_v
            
            dM = torch.bmm(err.unsqueeze(-1), k.unsqueeze(1))
            M = alpha * M + beta * dM
            
            fused = torch.cat([h_in, r], dim=-1)
            h_out = torch.tanh(self.out_proj(fused))
            outputs.append(self.head(h_out))
        return torch.stack(outputs, dim=1)


# =========================================================================
# 5. DATASET GENERATION: 3D PRECESSION & SO(3) ROTOR TRAJECTORY
# =========================================================================
def generate_rotor_dynamics(batch_size=64, seq_len=20):
    """
    Simulates non-commutative 3D spinning top / gyroscope precession dynamics:
    Rotor q(t) follows d/dt q = 0.5 * q (x) omega(t), where omega precesses in R^3.
    """
    X = torch.zeros(batch_size, seq_len, 4)
    Y = torch.zeros(batch_size, seq_len, 4)
    
    dt = 0.08
    for b in range(batch_size):
        # Random initial orientation q0 in S^3
        q = np.random.randn(4)
        q /= np.linalg.norm(q)
        
        # Precession frequencies
        w_spin = np.random.uniform(2.0, 4.0)
        w_prec = np.random.uniform(0.5, 1.5)
        phase = np.random.uniform(0, 2 * np.pi)
        
        for t in range(seq_len):
            X[b, t, :] = torch.tensor(q, dtype=torch.float32)
            
            # Instantaneous angular velocity in body frame
            t_sim = t * dt
            omega = np.array([
                np.sin(w_prec * t_sim + phase) * 0.8,
                np.cos(w_prec * t_sim + phase) * 0.8,
                w_spin
            ])
            
            # Incremental rotor dq = [cos(||omega|| dt / 2), (omega / ||omega||) sin(||omega|| dt / 2)]
            om_norm = np.linalg.norm(omega)
            theta = om_norm * dt
            axis = omega / om_norm
            dq = np.array([np.cos(theta / 2.0), axis[0]*np.sin(theta/2.0), axis[1]*np.sin(theta/2.0), axis[2]*np.sin(theta/2.0)])
            
            # Hamilton product q_next = q (x) dq
            qr, qi, qj, qk = q
            dqr, dqi, dqj, dqk = dq
            q_next = np.array([
                qr*dqr - qi*dqi - qj*dqj - qk*dqk,
                qr*dqi + qi*dqr + qj*dqk - qk*dqj,
                qr*dqj - qi*dqk + qj*dqr + qk*dqi,
                qr*dqk + qi*dqj - qj*dqi + qk*dqr
            ])
            q_next /= np.linalg.norm(q_next)
            
            Y[b, t, :] = torch.tensor(q_next, dtype=torch.float32)
            q = q_next
            
    return X, Y


# =========================================================================
# 6. EXECUTE EMPIRICAL BENCHMARK
# =========================================================================
def run_benchmark():
    print("=" * 80)
    print("EMPIRICAL BENCHMARK SHOWDOWN: H-DeltaNet vs TRANSFORMER vs REAL-DeltaNet")
    print("Dynamical Task: SO(3) Gyroscopic Precession Trajectory Forecasting")
    print("=" * 80)
    
    # 1. Models Initialization
    # HDeltaNet operates on 1 quaternion unit = 4 real numbers
    m_hdelta = HDeltaNet(in_q=1, d_model_q=4, d_key_q=4, d_val_q=4, out_q=1)
    # Transformer operates on 4 real numbers with d_model=16
    m_trans  = CausalTransformer(in_dim=4, d_model=16, nhead=4, num_layers=2, out_dim=4)
    # Real DeltaNet operates on 4 real numbers with d_model=16, d_key=16, d_val=16
    m_rdelta = RealDeltaNet(in_dim=4, d_model=16, d_key=16, d_val=16, out_dim=4)
    
    # Count parameters
    params_hdelta = sum(p.numel() for p in m_hdelta.parameters())
    params_trans  = sum(p.numel() for p in m_trans.parameters())
    params_rdelta = sum(p.numel() for p in m_rdelta.parameters())
    
    print(f"Model Complexity & Degrees of Freedom:")
    print(f" - [H-DeltaNet]       Parameters: {params_hdelta:5d}  (Quaternion Hamilton algebra)")
    print(f" - [Causal Transformer] Parameters: {params_trans:5d}  (Multi-Head Self-Attention)")
    print(f" - [Real DeltaNet]    Parameters: {params_rdelta:5d}  (Unconstrained Real Linear Attention)")
    print("-" * 80)
    
    # Optimizers
    opt_hdelta = torch.optim.AdamW(m_hdelta.parameters(), lr=0.008, weight_decay=1e-4)
    opt_trans  = torch.optim.AdamW(m_trans.parameters(), lr=0.008, weight_decay=1e-4)
    opt_rdelta = torch.optim.AdamW(m_rdelta.parameters(), lr=0.008, weight_decay=1e-4)
    
    # Geodesic loss function on S^3
    def geodesic_loss(pred, target):
        pred_norm = quat_normalize(pred)
        # 1 - |<q_pred, q_true>|
        inner = torch.abs(torch.sum(pred_norm * target, dim=-1))
        return torch.mean(1.0 - inner)

    epochs = 40
    batch_size = 64
    seq_len = 20
    
    start_train = time.time()
    for ep in range(1, epochs + 1):
        X, Y = generate_rotor_dynamics(batch_size, seq_len)
        X_h = X.unsqueeze(2) # (B, T, 1, 4) for quaternion layer
        
        # 1. Train H-DeltaNet
        opt_hdelta.zero_grad()
        out_h = m_hdelta(X_h).squeeze(2)
        loss_h = geodesic_loss(out_h, Y)
        loss_h.backward()
        torch.nn.utils.clip_grad_norm_(m_hdelta.parameters(), 1.0)
        opt_hdelta.step()
        
        # 2. Train Transformer
        opt_trans.zero_grad()
        out_t = m_trans(X)
        loss_t = geodesic_loss(out_t, Y)
        loss_t.backward()
        torch.nn.utils.clip_grad_norm_(m_trans.parameters(), 1.0)
        opt_trans.step()
        
        # 3. Train Real DeltaNet
        opt_rdelta.zero_grad()
        out_r = m_rdelta(X)
        loss_r = geodesic_loss(out_r, Y)
        loss_r.backward()
        torch.nn.utils.clip_grad_norm_(m_rdelta.parameters(), 1.0)
        opt_rdelta.step()
        
        if ep % 10 == 0 or ep == epochs:
            print(f"Epoch {ep:2d}/{epochs} | "
                  f"H-DeltaNet Loss: {loss_h.item():.4f} | "
                  f"Transformer Loss: {loss_t.item():.4f} | "
                  f"Real-DeltaNet Loss: {loss_r.item():.4f}")
                  
    total_train_time = time.time() - start_train
    print("-" * 80)
    print(f"Training finalized in {total_train_time:.2f}s.")
    
    # =====================================================================
    # 7. HOLDOUT TEST & INFERENCE BENCHMARK (N=256 trajectories)
    # =====================================================================
    print("\n" + "=" * 80)
    print("HOLDOUT EVALUATION & INFERENCE LATENCY BENCHMARK (N=256 trajectories)")
    print("=" * 80)
    
    with torch.no_grad():
        X_test, Y_test = generate_rotor_dynamics(256, seq_len)
        X_test_h = X_test.unsqueeze(2)
        
        # 1. H-DeltaNet Evaluation
        t0 = time.perf_counter()
        pred_h = quat_normalize(m_hdelta(X_test_h).squeeze(2))
        lat_h = (time.perf_counter() - t0) * 1000.0
        fid_h = torch.abs(torch.sum(pred_h * Y_test, dim=-1)).mean().item()
        ang_h = 2.0 * np.arccos(np.clip(fid_h, -1.0, 1.0)) * (180.0 / np.pi)
        
        # 2. Transformer Evaluation
        t0 = time.perf_counter()
        pred_t = quat_normalize(m_trans(X_test))
        lat_t = (time.perf_counter() - t0) * 1000.0
        fid_t = torch.abs(torch.sum(pred_t * Y_test, dim=-1)).mean().item()
        ang_t = 2.0 * np.arccos(np.clip(fid_t, -1.0, 1.0)) * (180.0 / np.pi)
        
        # 3. Real DeltaNet Evaluation
        t0 = time.perf_counter()
        pred_r = quat_normalize(m_rdelta(X_test))
        lat_r = (time.perf_counter() - t0) * 1000.0
        fid_r = torch.abs(torch.sum(pred_r * Y_test, dim=-1)).mean().item()
        ang_r = 2.0 * np.arccos(np.clip(fid_r, -1.0, 1.0)) * (180.0 / np.pi)
        
    print(f"{'Architecture':<22} | {'Params':<8} | {'SO(3) Fidelity':<15} | {'Angular Error':<14} | {'Batch Latency':<12}")
    print("-" * 80)
    print(f"{'H-DeltaNet (Ours)':<22} | {params_hdelta:<8d} | {fid_h*100:6.2f}%         | {ang_h:6.2f}°       | {lat_h:6.2f} ms")
    print(f"{'Causal Transformer':<22} | {params_trans:<8d} | {fid_t*100:6.2f}%         | {ang_t:6.2f}°       | {lat_t:6.2f} ms")
    print(f"{'Real-Valued DeltaNet':<22} | {params_rdelta:<8d} | {fid_r*100:6.2f}%         | {ang_r:6.2f}°       | {lat_r:6.2f} ms")
    print("-" * 80)
    
    # Efficiency ratio
    comp_ratio = params_trans / params_hdelta
    print(f"\n[EMPIRICAL VERDICT]")
    print(f"H-DeltaNet delivers {fid_h*100:.2f}% SO(3) Rotor Fidelity with only {params_hdelta} parameters.")
    print(f"It achieves an angular error of {ang_h:.2f}°, while compressing parameters by {comp_ratio:.1f}x vs Transformer!")
    print(f"Recurrent tensor state O(1) step inference avoids O(T) KV-cache explosion during deployment.")
    print("=" * 80)

if __name__ == "__main__":
    run_benchmark()
