"""
Layer-Wise Memory Tensor Architecture vs Standard Vector RNN
Empirical verification of per-layer associative memory tensors on delayed key-value recall.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

torch.manual_seed(42)
np.random.seed(42)

class TensorMemoryLayer(nn.Module):
    """
    Recurrent layer with an explicit rank-2 Associative Memory Tensor M_t in R^{d_val x d_key}.
    Updates via dynamic Delta-rule outer product:
    M_t = alpha_t * M_{t-1} + beta_t * (v_t - M_{t-1} k_t) (x) k_t^T
    Reads via:
    r_t = M_{t-1} q_t
    """
    def __init__(self, d_in, d_key, d_val, d_out, use_tensor_memory=True):
        super().__init__()
        self.d_key = d_key
        self.d_val = d_val
        self.d_out = d_out
        self.use_tensor_memory = use_tensor_memory

        self.W_q = nn.Linear(d_in, d_key, bias=False)
        self.W_k = nn.Linear(d_in, d_key, bias=False)
        self.W_v = nn.Linear(d_in, d_val, bias=False)
        
        # Retention (decay) gate and write rate gate
        self.gate_alpha = nn.Linear(d_in, 1) # decay lambda in [0, 1]
        self.gate_beta = nn.Linear(d_in, 1)  # write rate in [0, 1]
        
        # Recurrent state projection if baseline
        self.W_rec = nn.Linear(d_val, d_val) if not use_tensor_memory else None
        
        # Readout projection
        read_dim = d_val if use_tensor_memory else d_val
        self.out_proj = nn.Linear(d_in + read_dim, d_out)
        self.ln = nn.LayerNorm(d_out)

    def forward(self, x_seq):
        """
        x_seq: (B, T, d_in)
        Returns:
            out_seq: (B, T, d_out)
            final_M: (B, d_val, d_key)
        """
        B, T, _ = x_seq.shape
        device = x_seq.device
        
        if self.use_tensor_memory:
            # Memory Tensor per batch element: (B, d_val, d_key)
            M = torch.zeros(B, self.d_val, self.d_key, device=device)
        else:
            # Standard hidden vector state: (B, d_val)
            h = torch.zeros(B, self.d_val, device=device)
            
        outputs = []
        
        for t in range(T):
            xt = x_seq[:, t, :] # (B, d_in)
            
            if self.use_tensor_memory:
                # 1. Project keys, queries, values
                q = F.normalize(self.W_q(xt), dim=-1) # (B, d_key)
                k = F.normalize(self.W_k(xt), dim=-1) # (B, d_key)
                v = self.W_v(xt)                     # (B, d_val)
                
                alpha = torch.sigmoid(self.gate_alpha(xt)).unsqueeze(-1) # (B, 1, 1)
                beta = torch.sigmoid(self.gate_beta(xt)).unsqueeze(-1)   # (B, 1, 1)
                
                # 2. Read from previous memory tensor: r_t = M_{t-1} @ q_t
                # M: (B, d_val, d_key), q: (B, d_key, 1) -> r: (B, d_val)
                r = torch.bmm(M, q.unsqueeze(-1)).squeeze(-1)
                
                # 3. Delta-rule error update: e = v_t - M_{t-1} @ k_t
                pred_v = torch.bmm(M, k.unsqueeze(-1)).squeeze(-1)
                err = v - pred_v # (B, d_val)
                
                # 4. Associative Tensor Write: dM = err (x) k^T
                dM = torch.bmm(err.unsqueeze(-1), k.unsqueeze(1)) # (B, d_val, d_key)
                M = alpha * M + beta * dM
                
                # 5. Output fusion
                fused = torch.cat([xt, r], dim=-1)
            else:
                # Standard Vector RNN baseline
                v = self.W_v(xt)
                h = torch.tanh(self.W_rec(h) + v)
                fused = torch.cat([xt, h], dim=-1)
                
            out = self.ln(F.gelu(self.out_proj(fused)))
            outputs.append(out)
            
        return torch.stack(outputs, dim=1), (M if self.use_tensor_memory else h)


class MultiLayerTensorMemoryNet(nn.Module):
    """Deep network where EVERY layer holds its own persistent memory tensor."""
    def __init__(self, d_in, d_model, d_key, d_val, num_layers=3, use_tensor_memory=True):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        
        current_in = d_in
        for i in range(num_layers):
            layer = TensorMemoryLayer(
                d_in=current_in,
                d_key=d_key,
                d_val=d_val,
                d_out=d_model,
                use_tensor_memory=use_tensor_memory
            )
            self.layers.append(layer)
            current_in = d_model
            
        self.head = nn.Linear(d_model, d_val)

    def forward(self, x):
        h = x
        for layer in self.layers:
            h, _ = layer(h)
        return self.head(h)


def generate_delayed_associative_data(batch_size=64, num_kv_pairs=4, delay_steps=25, d_key=16, d_val=16):
    """
    Stress test: Store multiple (Key, Value) pairs in memory, endure delay_steps of distraction noise,
    then query one of the stored keys. Target is the exact matching Value.
    """
    d_in = d_key + d_val + 2 # Key channel, Value channel, [is_store_flag, is_query_flag]
    
    # Store steps: num_kv_pairs
    # Delay steps: delay_steps (zeros / noise)
    # Query step: 1 step
    T = num_kv_pairs + delay_steps + 1
    
    X = torch.zeros(batch_size, T, d_in)
    Y = torch.zeros(batch_size, d_val)
    
    for b in range(batch_size):
        # Generate orthogonal/random keys and values
        keys = F.normalize(torch.randn(num_kv_pairs, d_key), dim=-1)
        values = torch.randn(num_kv_pairs, d_val)
        
        # 1. Store phase
        for i in range(num_kv_pairs):
            X[b, i, :d_key] = keys[i]
            X[b, i, d_key:d_key+d_val] = values[i]
            X[b, i, -2] = 1.0 # Store flag
            
        # 2. Delay phase (noise distraction)
        for d in range(delay_steps):
            t = num_kv_pairs + d
            X[b, t, :d_key] = 0.05 * torch.randn(d_key) # low noise
            X[b, t, -2:] = 0.0 # no flags
            
        # 3. Query phase: pick a random key from the past
        query_idx = np.random.randint(0, num_kv_pairs)
        t_query = T - 1
        X[b, t_query, :d_key] = keys[query_idx]
        X[b, t_query, -1] = 1.0 # Query flag
        
        # Target is the associated value
        Y[b] = values[query_idx]
        
    return X, Y


def run_experiment():
    print("=" * 70)
    print("HYPOTHESIS FALSIFICATION: LAYER-WISE TENSOR MEMORY VS STANDARD VECTOR RNN")
    print("Task: Delayed Multi-Key Associative Retrieval across 25 distraction steps")
    print("=" * 70)
    
    d_key = 16
    d_val = 16
    d_in = d_key + d_val + 2
    d_model = 32
    num_layers = 2
    delay_steps = 25
    num_pairs = 4
    
    # 1. Instantiate both models
    tensor_model = MultiLayerTensorMemoryNet(d_in, d_model, d_key, d_val, num_layers=num_layers, use_tensor_memory=True)
    vector_model = MultiLayerTensorMemoryNet(d_in, d_model, d_key, d_val, num_layers=num_layers, use_tensor_memory=False)
    
    opt_tensor = torch.optim.AdamW(tensor_model.parameters(), lr=0.005)
    opt_vector = torch.optim.AdamW(vector_model.parameters(), lr=0.005)
    
    epochs = 120
    batch_size = 64
    
    print(f"Configurations:")
    print(f" - Layers: {num_layers}")
    print(f" - Memory Tensor per layer: ({d_val} x {d_key}) = {d_val * d_key} storage cells per layer")
    print(f" - Delay Horizon: {delay_steps} steps | Stored Pairs: {num_pairs}")
    print("-" * 70)
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        # Generate batch
        X, Y = generate_delayed_associative_data(batch_size, num_pairs, delay_steps, d_key, d_val)
        
        # Train Tensor Memory Model
        opt_tensor.zero_grad()
        out_tensor = tensor_model(X)[:, -1, :] # query at final step
        loss_tensor = F.mse_loss(out_tensor, Y)
        loss_tensor.backward()
        torch.nn.utils.clip_grad_norm_(tensor_model.parameters(), 1.0)
        opt_tensor.step()
        
        # Train Standard Vector Model
        opt_vector.zero_grad()
        out_vector = vector_model(X)[:, -1, :]
        loss_vector = F.mse_loss(out_vector, Y)
        loss_vector.backward()
        torch.nn.utils.clip_grad_norm_(vector_model.parameters(), 1.0)
        opt_vector.step()
        
        if epoch % 20 == 0 or epoch == epochs:
            # Cosine similarity evaluation
            with torch.no_grad():
                cos_tensor = F.cosine_similarity(out_tensor, Y, dim=-1).mean().item()
                cos_vector = F.cosine_similarity(out_vector, Y, dim=-1).mean().item()
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Tensor M Loss: {loss_tensor.item():.4f} (Cos: {cos_tensor:+.3f}) | "
                  f"Vector RNN Loss: {loss_vector.item():.4f} (Cos: {cos_vector:+.3f})")
                  
    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"Training completed in {elapsed:.2f}s.")
    
    # Final evaluation on holdout test set
    with torch.no_grad():
        X_test, Y_test = generate_delayed_associative_data(256, num_pairs, delay_steps, d_key, d_val)
        pred_tensor = tensor_model(X_test)[:, -1, :]
        pred_vector = vector_model(X_test)[:, -1, :]
        
        test_mse_tensor = F.mse_loss(pred_tensor, Y_test).item()
        test_mse_vector = F.mse_loss(pred_vector, Y_test).item()
        
        sim_tensor = F.cosine_similarity(pred_tensor, Y_test, dim=-1).mean().item()
        sim_vector = F.cosine_similarity(pred_vector, Y_test, dim=-1).mean().item()
        
    print("\nHOLDOUT TEST RESULTS (N=256 sequences):")
    print(f"Layer-Wise Tensor Memory -> Test MSE: {test_mse_tensor:.4f} | Retrieval Cosine Sim: {sim_tensor:.4f}")
    print(f"Standard Vector RNN       -> Test MSE: {test_mse_vector:.4f} | Retrieval Cosine Sim: {sim_vector:.4f}")
    
    if test_mse_tensor < test_mse_vector and sim_tensor > 0.8:
        print("\n[VERDICT: HYPOTHESIS VALIDATED]")
        print("Layer-wise tensor memory preserves associative key-value bindings across the delay horizon.")
        print("Vector-state RNN suffered catastrophic forgetting / interference in the delay phase.")
    else:
        print("\n[VERDICT: HYPOTHESIS FALSIFIED]")
    print("=" * 70)

if __name__ == "__main__":
    run_experiment()
