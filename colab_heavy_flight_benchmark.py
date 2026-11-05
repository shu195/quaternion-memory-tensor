"""
Heavy Cloud GPU Monte-Carlo Benchmark: 6-DOF IMU Attitude Telemetry
Executed on Google Colab Cloud GPU via Colab CLI.

Mathematically stabilized:
- Smooth geodesic loss: 1 - <p, y>^2 (strictly C^inf, zero singularity at 0)
- Bounded memory tensor projection: preserves finite Frobenius energy over long horizons T=2000
- Gradient norm clipping to 1.0
"""

import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 80)
print("GOOGLE COLAB CLOUD GPU BENCHMARK HARNESS INITIALIZING")
print("=" * 80)

if torch.cuda.is_available():
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Accelerated Compute Device: {gpu_name} (CUDA Active)")
else:
    device = torch.device('cpu')
    print("Accelerated Compute Device: Local CPU")

np.random.seed(42)
torch.manual_seed(42)

# =========================================================================
# QUATERNION KINEMATICS
# =========================================================================
def quat_mult(q1, q2):
    r1, i1, j1, k1 = q1
    r2, i2, j2, k2 = q2
    return np.array([
        r1*r2 - i1*i2 - j1*j2 - k1*k2,
        r1*i2 + i1*r2 + j1*k2 - k1*j2,
        r1*j2 - i1*k2 + j1*r2 + k1*i2,
        r1*k2 + i1*j2 - j1*i2 + k1*r2
    ])

def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_normalize(q):
    return q / (np.linalg.norm(q) + 1e-12)

def rotate_vector_by_quat(v, q):
    q_v = np.array([0.0, v[0], v[1], v[2]])
    q_rot = quat_mult(quat_mult(q, q_v), quat_conj(q))
    return q_rot[1:4]

def angular_error_deg(q_pred, q_true):
    inner = abs(float(np.dot(quat_normalize(q_pred), quat_normalize(q_true))))
    inner = np.clip(inner, -1.0, 1.0)
    return 2.0 * np.arccos(inner) * (180.0 / np.pi)

# =========================================================================
# 6-DOF IMU FLIGHT TELEMETRY GENERATOR
# =========================================================================
def generate_flight_trajectory(steps=2000, dt=0.02, trajectory_id=0):
    np.random.seed(trajectory_id + 100)
    q_true = np.array([1.0, 0.0, 0.0, 0.0])
    gyro_bias = np.random.uniform(-0.03, 0.03, size=3)
    
    t_vals = np.linspace(0, steps * dt, steps)
    q_history, gyro_meas_history, accel_meas_history = [], [], []
    
    f1, f2, f3 = np.random.uniform(0.8, 2.5, size=3)
    
    for t in t_vals:
        omega_true = np.array([
            1.8 * np.sin(f1 * t) + 0.8 * np.cos(2.2 * t),
            1.5 * np.cos(f2 * t) - 0.6 * np.sin(2.8 * t),
            0.9 * np.sin(f3 * t) + 0.4 * np.cos(1.4 * t)
        ])
        
        om_norm = np.linalg.norm(omega_true)
        if om_norm > 1e-8:
            axis = omega_true / om_norm
            theta = om_norm * dt
            dq = np.array([np.cos(theta / 2.0), axis[0]*np.sin(theta/2.0), axis[1]*np.sin(theta/2.0), axis[2]*np.sin(theta/2.0)])
            q_true = quat_normalize(quat_mult(q_true, dq))
            
        q_history.append(q_true.copy())
        
        gyro_bias += 0.0001 * np.random.randn(3)
        gyro_noise = 0.012 * np.random.randn(3)
        gyro_meas = omega_true + gyro_bias + gyro_noise
        gyro_meas_history.append(gyro_meas)
        
        g_body = rotate_vector_by_quat(np.array([0.0, 0.0, -9.81]), quat_conj(q_true))
        thrust_spike = np.array([
            0.8 * np.sin(2.0 * t),
            0.6 * np.cos(2.5 * t),
            2.5 * (1.0 + np.sin(1.0 * t))
        ])
        accel_noise = 0.06 * np.random.randn(3)
        accel_meas = g_body + thrust_spike + accel_noise
        accel_meas_history.append(accel_meas)
        
    return np.array(q_history), np.array(gyro_meas_history), np.array(accel_meas_history)

# =========================================================================
# ESTIMATORS
# =========================================================================
class MultiplicativeEKF:
    def __init__(self, dt=0.02):
        self.dt = dt
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.b = np.zeros(3)
        self.P = np.eye(6) * 0.1
        self.Q = np.eye(6)
        self.Q[0:3, 0:3] *= 0.001
        self.Q[3:6, 3:6] *= 0.0001
        self.R = np.eye(3) * 0.5

    def step(self, gyro_meas, accel_meas):
        unbiased_omega = gyro_meas - self.b
        om_norm = np.linalg.norm(unbiased_omega)
        if om_norm > 1e-8:
            axis = unbiased_omega / om_norm
            theta = om_norm * self.dt
            dq = np.array([np.cos(theta / 2.0), axis[0]*np.sin(theta/2.0), axis[1]*np.sin(theta/2.0), axis[2]*np.sin(theta/2.0)])
            self.q = quat_normalize(quat_mult(self.q, dq))
            
        wx, wy, wz = unbiased_omega
        om_skew = np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]])
        F_mat = np.eye(6)
        F_mat[0:3, 0:3] -= om_skew * self.dt
        F_mat[0:3, 3:6] = -np.eye(3) * self.dt
        self.P = F_mat @ self.P @ F_mat.T + self.Q
        
        v_ref = np.array([0.0, 0.0, -1.0])
        z_pred = rotate_vector_by_quat(v_ref, quat_conj(self.q))
        a_norm = np.linalg.norm(accel_meas)
        z_meas = accel_meas / a_norm if a_norm > 1e-4 else z_pred
        
        zx, zy, zz = z_pred
        z_skew = np.array([[0, -zz, zy], [zz, 0, -zx], [-zy, zx, 0]])
        H = np.zeros((3, 6))
        H[0:3, 0:3] = -z_skew
        
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ (z_meas - z_pred)
        
        d_theta = dx[0:3]
        if np.linalg.norm(d_theta) > 1e-8:
            d_q_corr = np.array([1.0, 0.5*d_theta[0], 0.5*d_theta[1], 0.5*d_theta[2]])
            self.q = quat_normalize(quat_mult(self.q, quat_normalize(d_q_corr)))
            
        self.b += dx[3:6]
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        return self.q.copy()

class MadgwickFilter:
    def __init__(self, dt=0.02, beta=0.08):
        self.dt = dt
        self.beta = beta
        self.q = np.array([1.0, 0.0, 0.0, 0.0])

    def step(self, gyro_meas, accel_meas):
        q = self.q
        gx, gy, gz = gyro_meas
        qDot1 = 0.5 * (-q[1]*gx - q[2]*gy - q[3]*gz)
        qDot2 = 0.5 * ( q[0]*gx + q[2]*gz - q[3]*gy)
        qDot3 = 0.5 * ( q[0]*gy - q[1]*gz + q[3]*gx)
        qDot4 = 0.5 * ( q[0]*gz + q[1]*gy - q[2]*gx)
        
        a_norm = np.linalg.norm(accel_meas)
        if a_norm > 1e-4:
            ax, ay, az = accel_meas / a_norm
            f1 = 2.0*(q[1]*q[3] - q[0]*q[2]) - ax
            f2 = 2.0*(q[0]*q[1] + q[2]*q[3]) - ay
            f3 = 2.0*(0.5 - q[1]**2 - q[2]**2) - az
            
            J_11or24 = 2.0*q[2]; J_12or23 = 2.0*q[3]; J_13or22 = 2.0*q[0]; J_14or21 = 2.0*q[1]; J_32 = 2.0*J_14or21; J_33 = 2.0*J_11or24
            step1 = -J_11or24*f1 + J_13or22*f2
            step2 =  J_12or23*f1 + J_14or21*f2 - J_32*f3
            step3 = -J_13or22*f1 + J_12or23*f2 - J_33*f3
            step4 =  J_14or21*f1 + J_11or24*f2
            step_norm = np.sqrt(step1**2 + step2**2 + step3**2 + step4**2) + 1e-12
            
            qDot1 -= self.beta * (step1 / step_norm)
            qDot2 -= self.beta * (step2 / step_norm)
            qDot3 -= self.beta * (step3 / step_norm)
            qDot4 -= self.beta * (step4 / step_norm)
            
        self.q[0] += qDot1 * self.dt
        self.q[1] += qDot2 * self.dt
        self.q[2] += qDot3 * self.dt
        self.q[3] += qDot4 * self.dt
        self.q = quat_normalize(self.q)
        return self.q.copy()

class HDeltaNetIMUEstimator(nn.Module):
    def __init__(self, dt=0.02, d_model_q=4, d_key_q=4, d_val_q=2):
        super().__init__()
        self.dt = dt
        self.d_model_q = d_model_q
        self.d_key_q = d_key_q
        self.d_val_q = d_val_q
        
        self.W_in = nn.Parameter(torch.randn(d_model_q, 2, 4) * 0.05)
        self.W_q  = nn.Parameter(torch.randn(d_key_q, d_model_q, 4) * 0.05)
        self.W_k  = nn.Parameter(torch.randn(d_key_q, d_model_q, 4) * 0.05)
        self.W_v  = nn.Parameter(torch.randn(d_val_q, d_model_q, 4) * 0.05)
        
        self.gate_alpha = nn.Linear(4 * d_model_q, 1)
        nn.init.constant_(self.gate_alpha.bias, 4.0)
        self.gate_beta = nn.Linear(4 * d_model_q, 1)
        nn.init.constant_(self.gate_beta.bias, -1.0)
        
        self.bias_head = nn.Linear(4 * (d_model_q + d_val_q), 3)
        nn.init.zeros_(self.bias_head.weight)
        nn.init.zeros_(self.bias_head.bias)
        
        self.corr_head = nn.Linear(4 * (d_model_q + d_val_q), 3)
        nn.init.zeros_(self.corr_head.weight)
        nn.init.zeros_(self.corr_head.bias)

    def quat_mult_torch(self, q1, q2):
        r1, i1, j1, k1 = q1.unbind(-1)
        r2, i2, j2, k2 = q2.unbind(-1)
        r = r1*r2 - i1*i2 - j1*j2 - k1*k2
        i = r1*i2 + i1*r2 + j1*k2 - k1*j2
        j = r1*j2 - i1*k2 + j1*r2 + k1*i2
        k = r1*k2 + i1*j2 - j1*i2 + k1*r2
        return torch.stack([r, i, j, k], dim=-1)

    def forward(self, gyro_seq, accel_seq):
        B, T, _ = gyro_seq.shape
        M = torch.zeros(B, self.d_val_q, self.d_key_q, 4, device=gyro_seq.device)
        est_q_list = []
        curr_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=gyro_seq.device).repeat(B, 1)
        
        for t in range(T):
            gt = gyro_seq[:, t, :]
            at = accel_seq[:, t, :]
            
            h_g = self.quat_mult_torch(self.W_in[0, 0].view(1, 4), gt)
            h_a = self.quat_mult_torch(self.W_in[0, 1].view(1, 4), at)
            h = F.gelu(h_g + h_a).repeat(1, self.d_model_q, 1)
            h_flat = h.reshape(B, -1)
            
            q_k = F.normalize(h[:, :self.d_key_q, :], dim=-1, eps=1e-8)
            k_k = F.normalize(h[:, :self.d_key_q, :], dim=-1, eps=1e-8)
            v_k = h[:, :self.d_val_q, :]
            
            alpha = torch.sigmoid(self.gate_alpha(h_flat)).view(B, 1, 1, 1)
            beta = torch.sigmoid(self.gate_beta(h_flat)).view(B, 1, 1, 1)
            
            r = torch.sum(self.quat_mult_torch(M, q_k.unsqueeze(1)), dim=2)
            pred_v = torch.sum(self.quat_mult_torch(M, k_k.unsqueeze(1)), dim=2)
            err = v_k - pred_v
            
            k_conj = torch.stack([k_k[..., 0], -k_k[..., 1], -k_k[..., 2], -k_k[..., 3]], dim=-1)
            dM = self.quat_mult_torch(err.unsqueeze(2), k_conj.unsqueeze(1))
            
            # Bound energy growth over long horizons
            M = alpha * M + beta * dM
            M = torch.clamp(M, -10.0, 10.0)
            
            fused = torch.cat([h_flat, r.reshape(B, -1)], dim=-1)
            bias_est = torch.tanh(self.bias_head(fused)) * 0.1
            tilt_corr = torch.tanh(self.corr_head(fused)) * 0.05
            
            omega_unbiased = gt[:, 1:4] - bias_est + tilt_corr
            half_theta = 0.5 * omega_unbiased * self.dt
            dq = torch.cat([torch.ones(B, 1, device=gyro_seq.device), half_theta], dim=-1)
            dq_norm = F.normalize(dq, dim=-1, eps=1e-8)
            
            curr_q = F.normalize(self.quat_mult_torch(curr_q, dq_norm), dim=-1, eps=1e-8)
            est_q_list.append(curr_q)
            
        return torch.stack(est_q_list, dim=1)

# =========================================================================
# RUN HEAVY MONTE-CARLO SUITE (30 MISSIONS)
# =========================================================================
def run_cloud_monte_carlo():
    num_runs = 30
    steps_per_run = 2000
    dt = 0.02
    
    print(f"Executing Monte-Carlo Evaluation:")
    print(f" - Mission Count: {num_runs} independent flight trajectories")
    print(f" - Horizon per Mission: {steps_per_run} steps ({steps_per_run*dt:.1f}s)")
    print(f" - Cumulative Simulation Time: {num_runs * steps_per_run * dt:.1f} seconds ({num_runs * steps_per_run:,} steps)")
    print("-" * 80)
    
    model = HDeltaNetIMUEstimator(dt=dt).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Fast pre-calibration
    q_init, g_init, a_init = generate_flight_trajectory(steps=1000, dt=dt, trajectory_id=999)
    g_ch = torch.zeros(20, 50, 4, device=device)
    a_ch = torch.zeros(20, 50, 4, device=device)
    y_ch = torch.zeros(20, 50, 4, device=device)
    for c in range(20):
        g_ch[c, :, 1:4] = torch.tensor(g_init[c*50:(c+1)*50], dtype=torch.float32, device=device)
        a_ch[c, :, 1:4] = torch.tensor(a_init[c*50:(c+1)*50], dtype=torch.float32, device=device)
        y_ch[c, :, :]   = torch.tensor(q_init[c*50:(c+1)*50], dtype=torch.float32, device=device)
        
    print("Calibrating H-DeltaNet Lie-Algebraic Memory Weights...")
    for ep in range(15):
        opt.zero_grad()
        p = model(g_ch, a_ch)
        inner = torch.sum(p * y_ch, dim=-1)
        loss = torch.mean(1.0 - inner**2) # C^inf smooth geodesic loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
    print("H-DeltaNet calibration complete. Running Monte-Carlo evaluation loop...")
    
    ekf_means, madg_means, h_means = [], [], []
    ekf_drifts, madg_drifts, h_drifts = [], [], []
    
    start_time = time.time()
    
    for run in range(num_runs):
        q_true, gyro_m, accel_m = generate_flight_trajectory(steps_per_run, dt, trajectory_id=run)
        
        # 1. EKF
        ekf = MultiplicativeEKF(dt=dt)
        q_ekf = [ekf.step(gyro_m[t], accel_m[t]) for t in range(steps_per_run)]
        err_ekf = [angular_error_deg(q_ekf[t], q_true[t]) for t in range(steps_per_run)]
        ekf_means.append(np.mean(err_ekf))
        ekf_drifts.append(err_ekf[-1])
        
        # 2. Madgwick
        madg = MadgwickFilter(dt=dt)
        q_madg = [madg.step(gyro_m[t], accel_m[t]) for t in range(steps_per_run)]
        err_madg = [angular_error_deg(q_madg[t], q_true[t]) for t in range(steps_per_run)]
        madg_means.append(np.mean(err_madg))
        madg_drifts.append(err_madg[-1])
        
        # 3. H-DeltaNet
        g_t = torch.zeros(1, steps_per_run, 4, device=device)
        a_t = torch.zeros(1, steps_per_run, 4, device=device)
        g_t[0, :, 1:4] = torch.tensor(gyro_m, dtype=torch.float32, device=device)
        a_t[0, :, 1:4] = torch.tensor(accel_m, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            q_h = model(g_t, a_t)[0].cpu().numpy()
        err_h = [angular_error_deg(q_h[t], q_true[t]) for t in range(steps_per_run)]
        h_means.append(np.mean(err_h))
        h_drifts.append(err_h[-1])
        
        if (run + 1) % 5 == 0 or run == 0:
            print(f"Mission {run+1:2d}/{num_runs} | "
                  f"H-DeltaNet Mean: {np.mean(err_h):5.2f}° (End Drift: {err_h[-1]:5.2f}°) | "
                  f"Madgwick Mean: {np.mean(err_madg):5.2f}° | "
                  f"EKF Mean: {np.mean(err_ekf):5.2f}°")
                  
    total_time = time.time() - start_time
    print("-" * 80)
    print(f"Completed {num_runs} missions ({num_runs*steps_per_run:,} steps) in {total_time:.2f}s.")
    
    print("\n" + "=" * 80)
    print(f"MONTE-CARLO STATISTICAL AUDIT (N = {num_runs} INDEPENDENT FLIGHT MISSIONS)")
    print("=" * 80)
    print(f"{'Estimator Architecture':<26} | {'Mean Error':<14} | {'Median Error':<14} | {'Terminal 40s Drift':<18}")
    print("-" * 80)
    print(f"{'H-DeltaNet (Ours)':<26} | {np.mean(h_means):6.2f}° ± {np.std(h_means):4.2f}° | {np.median(h_means):6.2f}°        | {np.mean(h_drifts):6.2f}° ± {np.std(h_drifts):4.2f}°")
    print(f"{'Madgwick Filter':<26} | {np.mean(madg_means):6.2f}° ± {np.std(madg_means):4.2f}° | {np.median(madg_means):6.2f}°        | {np.mean(madg_drifts):6.2f}° ± {np.std(madg_drifts):4.2f}°")
    print(f"{'Extended Kalman Filter':<26} | {np.mean(ekf_means):6.2f}° ± {np.std(ekf_means):4.2f}° | {np.median(ekf_means):6.2f}°        | {np.mean(ekf_drifts):6.2f}° ± {np.std(ekf_drifts):4.2f}°")
    print("=" * 80)

if __name__ == "__main__":
    run_cloud_monte_carlo()
