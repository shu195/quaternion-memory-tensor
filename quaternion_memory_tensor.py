"""
Quaternion Associative Memory Tensor (H-DeltaNet) From Scratch
Unifying Non-Commutative Quaternion Algebra (H) with Rank-2 Layer-Wise Memory Tensors.

Mathematical core:
Memory M in H^{d_v x d_k}.
Write: dM = (v - M (x) k) (x) k*
Read:  r = M (x) q
Where k* is the quaternion conjugate, guaranteeing (v (x) k*) (x) k = v * ||k||^2 = v!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

torch.manual_seed(42)
np.random.seed(42)

def quat_mult(q1, q2):
    """
    Hamilton product of two quaternion tensors q1, q2 of shape (..., 4).
    Components: [r, i, j, k]
    """
    r1, i1, j1, k1 = q1.unbind(-1)
    r2, i2, j2, k2 = q2.unbind(-1)
    
    r = r1*r2 - i1*i2 - j1*j2 - k1*k2
    i = r1*i2 + i1*r2 + j1*k2 - k1*j2
    j = r1*j2 - i1*k2 + j1*r2 + k1*i2
    k = r1*k2 + i1*j2 - j1*i2 + k1*r2
    return torch.stack([r, i, j, k], dim=-1)

def quat_conj(q):
    """Quaternion conjugate q* = [r, -i, -j, -k]."""
    r, i, j, k = q.unbind(-1)
    return torch.stack([r, -i, -j, -k], dim=-1)

def quat_norm(q, eps=1e-8):
    """Euclidean norm ||q|| = sqrt(r^2 + i^2 + j^2 + k^2)."""
    return torch.sqrt(torch.sum(q**2, dim=-1, keepdim=True) + eps)

def quat_normalize(q, eps=1e-8):
    """Projects quaternion to S^3 unit sphere."""
    return q / quat_norm(q, eps)

class QuaternionLinear(nn.Module):
    """Quaternion Linear layer using 4 real weight matrices (A, B, C, D)."""
    def __init__(self, in_q_features, out_q_features, bias=True):
        super().__init__()
        self.in_q = in_q_features
        self.out_q = out_q_features
        
        # Quaternion Glorot initialization
        std = 1.0 / np.sqrt(2.0 * in_q_features)
        self.A = nn.Parameter(torch.randn(out_q_features, in_q_features) * (std / 2.0))
        self.B = nn.Parameter(torch.randn(out_q_features, in_q_features) * (std / 2.0))
        self.C = nn.Parameter(torch.randn(out_q_features, in_q_features) * (std / 2.0))
        self.D = nn.Parameter(torch.randn(out_q_features, in_q_features) * (std / 2.0))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_q_features, 4))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        """
        x: (..., in_q, 4)
        Returns: (..., out_q, 4)
        """
        # Split input into components
        xr, xi, xj, xk = x.unbind(-1) # each (..., in_q)
        
        # Hamilton matrix multiplication
        yr = F.linear(xr, self.A) - F.linear(xi, self.B) - F.linear(xj, self.C) - F.linear(xk, self.D)
        yi = F.linear(xr, self.B) + F.linear(xi, self.A) - F.linear(xj, self.D) + F.linear(xk, self.C)
        yj = F.linear(xr, self.C) + F.linear(xi, self.D) + F.linear(xj, self.A) - F.linear(xk, self.B)
        yk = F.linear(xr, self.D) - F.linear(xi, self.C) + F.linear(xj, self.B) + F.linear(xk, self.A)
        
        out = torch.stack([yr, yi, yj, yk], dim=-1)
        if self.bias is not None:
            out = out + self.bias
        return out


class QuaternionMemoryTensorLayer(nn.Module):
    """
    Every layer holds a Quaternion Memory Tensor: M_t in H^{d_v x d_k}.
    Represented as a 4-channel tensor: (B, d_v, d_k, 4).
    """
    def __init__(self, in_q_dim, d_key_q, d_val_q, out_q_dim):
        super().__init__()
        self.d_key_q = d_key_q
        self.d_val_q = d_val_q
        
        self.W_q = QuaternionLinear(in_q_dim, d_key_q, bias=False)
        self.W_k = QuaternionLinear(in_q_dim, d_key_q, bias=False)
        self.W_v = QuaternionLinear(in_q_dim, d_val_q, bias=False)
        
        # Real-valued gates for retention (alpha) and write (beta)
        # Flattened quaternion input dimension = 4 * in_q_dim
        self.gate_alpha = nn.Linear(4 * in_q_dim, 1)
        nn.init.constant_(self.gate_alpha.bias, 4.0) # non-dissipative retention bias
        
        self.gate_beta = nn.Linear(4 * in_q_dim, 1)
        nn.init.constant_(self.gate_beta.bias, -0.5) # conservative write
        
        # Output fusion: combines input quaternion and recalled quaternion
        self.out_proj = QuaternionLinear(in_q_dim + d_val_q, out_q_dim)

    def quat_matvec(self, M, v):
        """
        Matrix-vector multiplication in H:
        M: (B, d_v, d_k, 4)
        v: (B, d_k, 4)
        Returns: (B, d_v, 4) where out_i = sum_j M_ij (x) v_j
        """
        B, dv, dk, _ = M.shape
        # Expand for broadcasting: M is (B, dv, dk, 4), v is (B, 1, dk, 4)
        v_exp = v.unsqueeze(1)
        # Compute elementwise Hamilton product M_ij (x) v_j
        prod = quat_mult(M, v_exp) # (B, dv, dk, 4)
        return torch.sum(prod, dim=2) # sum over d_k -> (B, dv, 4)

    def quat_outer(self, v, k_conj):
        """
        Outer product in H:
        v: (B, d_v, 4)
        k_conj: (B, d_k, 4)
        Returns: (B, d_v, d_k, 4) where out_ij = v_i (x) k_conj_j
        """
        v_exp = v.unsqueeze(2)        # (B, dv, 1, 4)
        k_exp = k_conj.unsqueeze(1)   # (B, 1, dk, 4)
        return quat_mult(v_exp, k_exp) # (B, dv, dk, 4)

    def forward(self, x_seq):
        """
        x_seq: (B, T, in_q_dim, 4)
        Returns: out_seq: (B, T, out_q_dim, 4), final_M: (B, d_val_q, d_key_q, 4)
        """
        B, T, in_q, _ = x_seq.shape
        device = x_seq.device
        
        # Persistent Quaternion Memory Tensor initialized to zero quaternion: [0, 0, 0, 0]
        M = torch.zeros(B, self.d_val_q, self.d_key_q, 4, device=device)
        outputs = []
        
        for t in range(T):
            xt = x_seq[:, t, :, :] # (B, in_q, 4)
            xt_flat = xt.reshape(B, 4 * in_q)
            
            # 1. Project Query, Key, Value in H
            q = quat_normalize(self.W_q(xt)) # (B, d_key_q, 4) on S^3
            k = quat_normalize(self.W_k(xt)) # (B, d_key_q, 4) on S^3
            v = self.W_v(xt)                # (B, d_val_q, 4)
            
            alpha = torch.sigmoid(self.gate_alpha(xt_flat)).view(B, 1, 1, 1)
            beta = torch.sigmoid(self.gate_beta(xt_flat)).view(B, 1, 1, 1)
            
            # 2. Associative Recall: r_t = M_{t-1} (x) q_t
            r = self.quat_matvec(M, q) # (B, d_val_q, 4)
            
            # 3. Delta-Rule Prediction: pred_v = M_{t-1} (x) k_t
            pred_v = self.quat_matvec(M, k)
            err = v - pred_v # error in H
            
            # 4. Quaternion Tensor Write: dM = err (x) k_t*
            k_conj = quat_conj(k)
            dM = self.quat_outer(err, k_conj) # (B, d_val_q, d_key_q, 4)
            
            # 5. Non-dissipative memory update
            M = alpha * M + beta * dM
            
            # 6. Fusion & output
            fused = torch.cat([xt, r], dim=1) # (B, in_q + d_val_q, 4)
            out = torch.tanh(self.out_proj(fused))
            outputs.append(out)
            
        return torch.stack(outputs, dim=1), M


class MultiLayerQuaternionMemoryNet(nn.Module):
    """Stack of Quaternion Memory Tensor Layers."""
    def __init__(self, in_q, d_model_q, d_key_q, d_val_q, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        curr_in = in_q
        for _ in range(num_layers):
            layer = QuaternionMemoryTensorLayer(curr_in, d_key_q, d_val_q, d_model_q)
            self.layers.append(layer)
            curr_in = d_model_q
            
        self.head = QuaternionLinear(d_model_q, d_val_q)

    def forward(self, x):
        h = x
        for layer in self.layers:
            h, _ = layer(h)
        return self.head(h)


def generate_so3_frame_recall_task(batch_size=64, num_landmarks=3, delay_steps=18):
    """
    Physical task: 3D Coordinate Frame Associative Memory.
    Store phase: Pairs of (Landmark_i in S^2, Rotor_i in SO(3)).
    Landmark is pure imaginary unit quaternion [0, kx, ky, kz].
    Rotor is unit quaternion [cos(theta/2), u * sin(theta/2)].
    Delay phase: delay_steps of distraction noise.
    Query phase: Given landmark K_target, recall the exact 3D rotor V_target!
    """
    in_q = 2 # 1 channel for Landmark key, 1 channel for Rotor value
    T = num_landmarks + delay_steps + 1
    
    X = torch.zeros(batch_size, T, in_q, 4)
    Y = torch.zeros(batch_size, 4) # Target unit rotor
    
    for b in range(batch_size):
        stored_keys = []
        stored_rotors = []
        
        for i in range(num_landmarks):
            # Pure imaginary unit quaternion key (3D point on S^2)
            k_vec = np.random.randn(3)
            k_vec /= np.linalg.norm(k_vec)
            key_q = torch.tensor([0.0, k_vec[0], k_vec[1], k_vec[2]], dtype=torch.float32)
            
            # 3D spatial rotation rotor in SO(3)
            axis = np.random.randn(3)
            axis /= np.linalg.norm(axis)
            theta = np.random.uniform(0.3, 2.5)
            rotor_q = torch.tensor([np.cos(theta / 2.0),
                                   axis[0] * np.sin(theta / 2.0),
                                   axis[1] * np.sin(theta / 2.0),
                                   axis[2] * np.sin(theta / 2.0)], dtype=torch.float32)
            
            stored_keys.append(key_q)
            stored_rotors.append(rotor_q)
            
            # Store phase
            X[b, i, 0, :] = key_q
            X[b, i, 1, :] = rotor_q
            
        # Delay phase: random micro-jitter
        for d in range(delay_steps):
            t = num_landmarks + d
            X[b, t, :, :] = 0.02 * torch.randn(in_q, 4)
            
        # Query phase: query random stored landmark
        q_idx = np.random.randint(0, num_landmarks)
        X[b, -1, 0, :] = stored_keys[q_idx]
        X[b, -1, 1, :] = 0.0 # No rotor presented! Must recall from memory!
        
        Y[b] = stored_rotors[q_idx]
        
    return X, Y


def run_experiment():
    print("=" * 75)
    print("QUATERNION ASSOCIATIVE MEMORY TENSOR (H-DeltaNet): SO(3) ROTOR RETRIEVAL")
    print("=" * 75)
    
    in_q = 2       # Landmark key + Rotor payload
    d_model_q = 4  # Hidden quaternion dimension (16 real components)
    d_key_q = 4    # Key quaternion dimension (16 real components)
    d_val_q = 1    # Output rotor quaternion (4 real components)
    num_layers = 2
    delay_steps = 18
    num_landmarks = 3
    
    model = MultiLayerQuaternionMemoryNet(in_q, d_model_q, d_key_q, d_val_q, num_layers=num_layers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.015, weight_decay=1e-4)
    
    epochs = 80
    batch_size = 64
    
    print(f"Model Specifications:")
    print(f" - Quaternion Layers: {num_layers}")
    print(f" - Memory Tensor per layer: M in H^{{{d_val_q} x {d_key_q}}} (stores non-commutative rotors)")
    print(f" - Delay Horizon: {delay_steps} distraction steps | Landmarks stored: {num_landmarks}")
    print("-" * 75)
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        X, Y = generate_so3_frame_recall_task(batch_size, num_landmarks, delay_steps)
        optimizer.zero_grad()
        
        # Forward pass: retrieve rotor at the query step
        out = model(X)[:, -1, 0, :] # (B, 4)
        out_norm = quat_normalize(out)
        
        # Loss: Geodesic inner product loss on S^3: 1 - |<q_pred, q_true>|
        # Because q and -q represent the identical 3D rotation in SO(3)!
        inner_prod = torch.abs(torch.sum(out_norm * Y, dim=-1))
        loss = torch.mean(1.0 - inner_prod)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if epoch % 10 == 0 or epoch == epochs:
            # Rotor fidelity = mean |<q_pred, q_true>| in [0, 1]
            fidelity = inner_prod.mean().item()
            # Mean angular error in degrees: 2 * arccos(fidelity)
            angular_err_deg = 2.0 * np.arccos(np.clip(fidelity, -1.0, 1.0)) * (180.0 / np.pi)
            print(f"Epoch {epoch:2d}/{epochs:2d} | Loss: {loss.item():.4f} | "
                  f"SO(3) Rotor Fidelity: {fidelity*100:.2f}% | Mean Angular Error: {angular_err_deg:.2f}°")
                  
    elapsed = time.time() - start_time
    print("-" * 75)
    print(f"Training completed in {elapsed:.2f}s.")
    
    # Holdout Evaluation on 256 unobserved 3D rotation trajectories
    with torch.no_grad():
        X_test, Y_test = generate_so3_frame_recall_task(256, num_landmarks, delay_steps)
        out_test = quat_normalize(model(X_test)[:, -1, 0, :])
        fidelity_test = torch.abs(torch.sum(out_test * Y_test, dim=-1)).mean().item()
        ang_err_test = 2.0 * np.arccos(np.clip(fidelity_test, -1.0, 1.0)) * (180.0 / np.pi)
        
    print("\nHOLDOUT TEST RESULTS (N=256 test sequences):")
    print(f"Mean SO(3) Rotor Fidelity: {fidelity_test*100:.2f}%")
    print(f"Mean Angular Error:         {ang_err_test:.2f}°")
    
    # Ground truth vs reconstructed rotor demonstration
    print("\nSAMPLE RECONSTRUCTION DEMO:")
    print(f"Target Rotor q_true: {np.round(Y_test[0].numpy(), 4)}")
    print(f"Recalled    q_pred: {np.round(out_test[0].numpy(), 4)}")
    print("=" * 75)

if __name__ == "__main__":
    run_experiment()
