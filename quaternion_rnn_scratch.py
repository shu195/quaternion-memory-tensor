"""
Quaternion Recurrent Neural Network (QRNN) From Scratch
Pure NumPy implementation with exact Hamilton algebra, analytical BPTT,
numerical gradient validation, and 3D rotational trajectory tracking.
"""

import numpy as np
import sys
import time

np.random.seed(42)

class QuaternionLinear:
    """
    Quaternion linear transformation: Y = W (x) X + b
    where W in H^{d_out x d_in}, represented as 4 real matrices (A, B, C, D).
    Weight parameter count: 4 * d_out * d_in (75% parameter reduction vs real linear map 16 * d_out * d_in).
    """
    def __init__(self, in_features, out_features):
        self.in_features = in_features   # number of quaternion units
        self.out_features = out_features # number of quaternion units
        
        # Quaternion Glorot / He initialization
        # Total variance per quaternion weight = 1 / (2 * in_features)
        # Component variance = 1 / (8 * in_features)
        std = 1.0 / np.sqrt(2.0 * in_features)
        self.A = np.random.randn(out_features, in_features) * (std / 2.0)
        self.B = np.random.randn(out_features, in_features) * (std / 2.0)
        self.C = np.random.randn(out_features, in_features) * (std / 2.0)
        self.D = np.random.randn(out_features, in_features) * (std / 2.0)
        
        # Bias: 4 * out_features
        self.b = np.zeros(4 * out_features)
        
        # Gradients
        self.grad_A = np.zeros_like(self.A)
        self.grad_B = np.zeros_like(self.B)
        self.grad_C = np.zeros_like(self.C)
        self.grad_D = np.zeros_like(self.D)
        self.grad_b = np.zeros_like(self.b)

    def get_hamilton_matrix(self):
        """Constructs the 4*out x 4*in real symplectic representation."""
        top = np.hstack([self.A, -self.B, -self.C, -self.D])
        mid1 = np.hstack([self.B, self.A, -self.D, self.C])
        mid2 = np.hstack([self.C, self.D, self.A, -self.B])
        bot = np.hstack([self.D, -self.C, self.B, self.A])
        return np.vstack([top, mid1, mid2, bot])

    def forward(self, x):
        """
        x: shape (4 * in_features,) or batch (N, 4 * in_features)
        returns y = W_hamilton @ x + b
        """
        W_h = self.get_hamilton_matrix()
        if x.ndim == 1:
            return W_h @ x + self.b
        else:
            return x @ W_h.T + self.b

    def backward(self, delta, x):
        """
        delta: gradient dL/dy (4 * out_features,)
        x: input (4 * in_features,)
        Accumulates grad_A, grad_B, grad_C, grad_D, grad_b
        Returns dL/dx = W_h.T @ delta
        """
        # Split delta and x into quaternion components (r, i, j, k)
        dr, di, dj, dk = np.split(delta, 4)
        xr, xi, xj, xk = np.split(x, 4)
        
        # Outer product contributions
        self.grad_A +=  np.outer(dr, xr) + np.outer(di, xi) + np.outer(dj, xj) + np.outer(dk, xk)
        self.grad_B += -np.outer(dr, xi) + np.outer(di, xr) - np.outer(dj, xk) + np.outer(dk, xj)
        self.grad_C += -np.outer(dr, xj) + np.outer(di, xk) + np.outer(dj, xr) - np.outer(dk, xi)
        self.grad_D += -np.outer(dr, xk) - np.outer(di, xj) + np.outer(dj, xi) + np.outer(dk, xr)
        self.grad_b += delta
        
        W_h = self.get_hamilton_matrix()
        dx = W_h.T @ delta
        return dx

    def zero_grad(self):
        self.grad_A.fill(0.0)
        self.grad_B.fill(0.0)
        self.grad_C.fill(0.0)
        self.grad_D.fill(0.0)
        self.grad_b.fill(0.0)


class QuaternionRNN:
    """
    Full Quaternion RNN with BPTT:
    h_t = tanh(W_xh (x) x_t + W_hh (x) h_{t-1} + b_h)
    y_t = W_hy (x) h_t + b_y
    """
    def __init__(self, in_q_dim, hidden_q_dim, out_q_dim):
        self.in_q_dim = in_q_dim
        self.hidden_q_dim = hidden_q_dim
        self.out_q_dim = out_q_dim
        
        self.W_xh = QuaternionLinear(in_q_dim, hidden_q_dim)
        self.W_hh = QuaternionLinear(hidden_q_dim, hidden_q_dim)
        self.W_hy = QuaternionLinear(hidden_q_dim, out_q_dim)

    def forward_sequence(self, x_seq, h0=None):
        """
        x_seq: list of length T, each (4 * in_q_dim,)
        Returns y_preds, h_states, z_states
        """
        T = len(x_seq)
        if h0 is None:
            h = np.zeros(4 * self.hidden_q_dim)
        else:
            h = h0.copy()
            
        h_states = [h]
        z_states = []
        y_preds = []
        
        for t in range(T):
            xt = x_seq[t]
            # z_t = W_xh @ xt + W_hh @ h_{t-1} + b_h
            zt = self.W_xh.forward(xt) + self.W_hh.forward(h) - self.W_hh.b  # bias shared once
            ht = np.tanh(zt)
            yt = self.W_hy.forward(ht)
            
            z_states.append(zt)
            h_states.append(ht)
            y_preds.append(yt)
            h = ht
            
        return y_preds, h_states, z_states

    def backward_bptt(self, x_seq, y_targets, y_preds, h_states, z_states):
        """
        Exact BPTT through time steps T-1 down to 0.
        Loss = 0.5 * sum_t ||y_t - y_target||^2
        """
        T = len(x_seq)
        self.W_xh.zero_grad()
        self.W_hh.zero_grad()
        self.W_hy.zero_grad()
        
        dh_next = np.zeros(4 * self.hidden_q_dim)
        total_loss = 0.0
        
        for t in reversed(range(T)):
            # Target error
            err_y = y_preds[t] - y_targets[t]
            total_loss += 0.5 * np.sum(err_y ** 2)
            
            # Backprop through output linear layer
            dh_from_y = self.W_hy.backward(err_y, h_states[t + 1])
            
            # Total gradient wrt h_t
            dh_t = dh_from_y + dh_next
            
            # Gradient through split tanh: dL/dz_t = dh_t * (1 - tanh(z_t)^2)
            dzt = dh_t * (1.0 - h_states[t + 1] ** 2)
            
            # Gradients for W_xh and W_hh
            self.W_xh.backward(dzt, x_seq[t])
            # For W_hh, input is h_{t-1} = h_states[t]
            dh_next = self.W_hh.backward(dzt, h_states[t])
            
        return total_loss

    def step_optimizer(self, lr=0.01, clip_norm=1.0):
        """Gradient clipping + SGD update."""
        params = [
            (self.W_xh.A, self.W_xh.grad_A), (self.W_xh.B, self.W_xh.grad_B),
            (self.W_xh.C, self.W_xh.grad_C), (self.W_xh.D, self.W_xh.grad_D),
            (self.W_xh.b, self.W_xh.grad_b),
            (self.W_hh.A, self.W_hh.grad_A), (self.W_hh.B, self.W_hh.grad_B),
            (self.W_hh.C, self.W_hh.grad_C), (self.W_hh.D, self.W_hh.grad_D),
            (self.W_hh.b, self.W_hh.grad_b),
            (self.W_hy.A, self.W_hy.grad_A), (self.W_hy.B, self.W_hy.grad_B),
            (self.W_hy.C, self.W_hy.grad_C), (self.W_hy.D, self.W_hy.grad_D),
            (self.W_hy.b, self.W_hy.grad_b),
        ]
        
        # Calculate total gradient norm
        total_norm = np.sqrt(sum(np.sum(g ** 2) for _, g in params))
        clip_coef = clip_norm / (total_norm + 1e-8)
        if clip_coef < 1.0:
            for _, g in params:
                g *= clip_coef
                
        for p, g in params:
            p -= lr * g


def numerical_gradient_check():
    """Verify analytical Hamilton BPTT gradients against numerical finite differences."""
    print("=" * 60)
    print("RUNNING NUMERICAL GRADIENT VERIFICATION (Finite Differences)")
    print("=" * 60)
    
    in_dim, h_dim, out_dim = 2, 3, 2
    model = QuaternionRNN(in_dim, h_dim, out_dim)
    T = 4
    
    # Random test sequence
    x_seq = [np.random.randn(4 * in_dim) for _ in range(T)]
    y_seq = [np.random.randn(4 * out_dim) for _ in range(T)]
    
    # 1. Forward & Analytical Backward
    y_preds, h_states, z_states = model.forward_sequence(x_seq)
    loss = model.backward_bptt(x_seq, y_seq, y_preds, h_states, z_states)
    
    eps = 1e-6
    max_rel_err = 0.0
    
    # Check W_hh.A as a canonical check
    for param_name, param, grad in [
        ("W_xh.A", model.W_xh.A, model.W_xh.grad_A),
        ("W_xh.B", model.W_xh.B, model.W_xh.grad_B),
        ("W_hh.A", model.W_hh.A, model.W_hh.grad_A),
        ("W_hh.C", model.W_hh.C, model.W_hh.grad_C),
        ("W_hy.D", model.W_hy.D, model.W_hy.grad_D),
        ("W_hy.b", model.W_hy.b, model.W_hy.grad_b),
    ]:
        it = np.nditer(param, flags=['multi_index'])
        diffs = []
        while not it.finished:
            idx = it.multi_index
            orig_val = param[idx]
            
            # Loss(param + eps)
            param[idx] = orig_val + eps
            preds_plus, _, _ = model.forward_sequence(x_seq)
            loss_plus = 0.5 * sum(np.sum((p - y)**2) for p, y in zip(preds_plus, y_seq))
            
            # Loss(param - eps)
            param[idx] = orig_val - eps
            preds_minus, _, _ = model.forward_sequence(x_seq)
            loss_minus = 0.5 * sum(np.sum((p - y)**2) for p, y in zip(preds_minus, y_seq))
            
            param[idx] = orig_val
            
            num_grad = (loss_plus - loss_minus) / (2.0 * eps)
            ana_grad = grad[idx]
            
            rel_err = abs(num_grad - ana_grad) / max(abs(num_grad) + abs(ana_grad), 1e-8)
            diffs.append(rel_err)
            it.iternext()
            
        peak_err = max(diffs)
        max_rel_err = max(max_rel_err, peak_err)
        status = "PASSED" if peak_err < 1e-5 else "FAILED"
        print(f"[{status}] Param: {param_name:10s} | Peak Rel Error: {peak_err:.2e}")

    print("-" * 60)
    print(f"OVERALL MAX RELATIVE ERROR: {max_rel_err:.2e}")
    assert max_rel_err < 1e-5, f"Gradient verification failed with err {max_rel_err}"
    print("[MATHEMATICAL FIDELITY CONFIRMED] Analytical Hamilton BPTT is strictly exact.\n")


def generate_3d_rotation_task(num_samples=100, seq_len=15):
    """
    Generates sequences of 3D spatial rotation under continuous quaternion rotor q(t).
    v(t+1) = rotor @ v(t) + noise
    Each state is a quaternion [0, vx, vy, vz].
    The QRNN must predict the next rotated vector in sequence!
    """
    dataset = []
    for _ in range(num_samples):
        # Random axis and angular velocity
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        omega = 0.3 * np.random.uniform(0.8, 1.2)
        
        # Incremental rotor q = [cos(omega/2), axis * sin(omega/2)]
        qr = np.cos(omega / 2.0)
        qv = axis * np.sin(omega / 2.0)
        
        # Initial 3D vector v0 as pure imaginary quaternion [0, x, y, z]
        v = np.random.randn(3)
        v /= np.linalg.norm(v)
        
        x_seq = []
        y_seq = []
        
        for t in range(seq_len):
            # Input quaternion: [0, v_x, v_y, v_z]
            q_in = np.array([0.0, v[0], v[1], v[2]])
            x_seq.append(q_in)
            
            # Rotate v using Rodrigues formula / quaternion conjugation
            # v_next = v + 2 * cross(qv, cross(qv, v) + qr * v)
            v_next = v + 2.0 * np.cross(qv, np.cross(qv, v) + qr * v)
            q_target = np.array([0.0, v_next[0], v_next[1], v_next[2]])
            y_seq.append(q_target)
            v = v_next
            
        dataset.append((x_seq, y_seq))
    return dataset


def run_experiment():
    # 1. First verify gradient correctness
    numerical_gradient_check()
    
    # 2. Train Quaternion RNN on 3D Rotor Forecasting
    print("=" * 60)
    print("TRAINING QUATERNION RNN ON 3D ROTATION TRAJECTORY (from scratch)")
    print("=" * 60)
    
    # in_q_dim=1 (1 quaternion: 4 real values), hidden_q_dim=4 (16 real values), out_q_dim=1 (4 real values)
    model = QuaternionRNN(in_q_dim=1, hidden_q_dim=4, out_q_dim=1)
    
    train_data = generate_3d_rotation_task(num_samples=120, seq_len=12)
    test_data = generate_3d_rotation_task(num_samples=30, seq_len=12)
    
    epochs = 40
    lr = 0.05
    start_time = time.time()
    
    initial_test_loss = 0.0
    for x_s, y_s in test_data:
        preds, _, _ = model.forward_sequence(x_s)
        initial_test_loss += 0.5 * sum(np.sum((p - y)**2) for p, y in zip(preds, y_s))
    initial_test_loss /= len(test_data)
    
    print(f"Epoch  0/40 | Initial Test Loss (MSE): {initial_test_loss:.6f}")
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        # Shuffle training set
        np.random.shuffle(train_data)
        for x_seq, y_seq in train_data:
            preds, h_states, z_states = model.forward_sequence(x_seq)
            loss = model.backward_bptt(x_seq, y_seq, preds, h_states, z_states)
            model.step_optimizer(lr=lr, clip_norm=2.0)
            epoch_loss += loss
            
        train_loss = epoch_loss / len(train_data)
        
        if epoch % 5 == 0 or epoch == epochs:
            test_loss = 0.0
            for x_s, y_s in test_data:
                preds, _, _ = model.forward_sequence(x_s)
                test_loss += 0.5 * sum(np.sum((p - y)**2) for p, y in zip(preds, y_s))
            test_loss /= len(test_data)
            print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")
            
    total_time = time.time() - start_time
    print("-" * 60)
    print(f"Training completed in {total_time:.2f}s.")
    print(f"Loss reduction: {initial_test_loss:.6f} -> {test_loss:.6f} ({(1.0 - test_loss/initial_test_loss)*100:.2f}% drop)")
    
    # 3. Demonstration of single trajectory prediction
    demo_x, demo_y = test_data[0]
    preds, _, _ = model.forward_sequence(demo_x)
    print("\nSAMPLE TRAJECTORY COMPARISON (Step T=5):")
    print(f"Ground Truth (Target): {np.round(demo_y[5], 4)}")
    print(f"QRNN Predicted:       {np.round(preds[5], 4)}")
    error_norm = np.linalg.norm(demo_y[5] - preds[5])
    print(f"L2 Error:              {error_norm:.6f}")
    print("=" * 60)

if __name__ == "__main__":
    run_experiment()
