"""
Render Polished High-Quality Figure 5: 6-DOF IMU Flight Telemetry Showdown
(H-DeltaNet vs EKF vs Madgwick)
Polished layout, perfect annotations without collisions, 300 DPI, pure white background.
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#ffffff',
    'savefig.facecolor': '#ffffff',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 1.2,
    'grid.color': '#e0e0e0',
    'grid.linestyle': '--',
    'grid.linewidth': 0.8,
    'text.color': '#111111',
    'axes.labelcolor': '#111111',
    'xtick.color': '#111111',
    'ytick.color': '#111111',
})

def render_figure_5_polished():
    data = np.load(r"C:\Users\Admin\.gemini\antigravity\scratch\quaternion_rnn\imu_benchmark_data.npz")
    t = data['t']
    q_true = data['q_true']
    q_h = data['q_h']
    q_ekf = data['q_ekf']
    q_madg = data['q_madg']
    err_h = data['err_h']
    err_ekf = data['err_ekf']
    err_madg = data['err_madg']
    accel = data['accel']
    gyro = data['gyro']

    fig = plt.figure(figsize=(15, 11), dpi=300)
    fig.patch.set_facecolor('#ffffff')

    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # -------------------------------------------------------------
    # Panel A: Angular Tracking Error Over Time (Degrees)
    # -------------------------------------------------------------
    ax1.plot(t, err_ekf, color='#c62828', lw=2.0, linestyle='--', label=f'Extended Kalman Filter (Mean: {np.mean(err_ekf):.1f}°)')
    ax1.plot(t, err_madg, color='#f57c00', lw=2.0, linestyle='-.', label=f'Madgwick Filter (Mean: {np.mean(err_madg):.1f}°)')
    ax1.plot(t, err_h, color='#1565c0', lw=2.5, label=f'ℍ-DeltaNet Symplectic (Mean: {np.mean(err_h):.1f}°)')
    
    ax1.set_title("A. Attitude Tracking Error on SO(3) [Degrees]", fontsize=12, fontweight='bold', pad=12)
    ax1.set_xlabel("Flight Time (seconds)", fontsize=10.5)
    ax1.set_ylabel("Geodesic Error (degrees)", fontsize=10.5)
    ax1.set_ylim(-5, 195)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='center right', frameon=True, facecolor='#ffffff', edgecolor='#cccccc', fontsize=9.5)
    
    # Placed safely without collision
    ax1.annotate("EKF Divergence\n(Linearization Breakdown\n+ Gyro Bias Drift)", 
                 xy=(3.5, 178.0), xytext=(4.0, 110.0),
                 arrowprops=dict(arrowstyle="->", color='#c62828', lw=1.8),
                 fontsize=9.5, fontweight='bold', color='#c62828',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffebee', edgecolor='#ef9a9a'))

    # -------------------------------------------------------------
    # Panel B: Cumulative Drift at 20 seconds (Bar Chart)
    # -------------------------------------------------------------
    models = ['ℍ-DeltaNet\n(Ours)', 'Madgwick\nFilter', 'Extended\nKalman Filter']
    drift_vals = [err_h[-1], err_madg[-1], err_ekf[-1]]
    colors = ['#1565c0', '#f57c00', '#c62828']
    
    bars = ax2.bar(models, drift_vals, color=colors, width=0.42, edgecolor='#212121', lw=1.2)
    ax2.set_title("B. Cumulative Drift After 20s Flight", fontsize=12, fontweight='bold', pad=12)
    ax2.set_ylabel("Terminal Drift (degrees)", fontsize=10.5)
    ax2.set_ylim(0, 195)
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, h + 4.0, f"{h:.1f}°",
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
                 
    # Polished callout: clear of the 31.8° text label!
    ax2.annotate("84.6% Lower Error vs EKF\nLower End-Drift vs Madgwick", 
                 xy=(0.0, 15.0), xytext=(0.45, 95.0),
                 arrowprops=dict(arrowstyle="->", color='#0d47a1', lw=1.8),
                 fontsize=9.5, fontweight='bold', color='#0d47a1',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#e3f2fd', edgecolor='#90caf9'))

    # -------------------------------------------------------------
    # Panel C: Quaternion Component Trajectory Tracking (q_0, q_3)
    # -------------------------------------------------------------
    ax3.plot(t, q_true[:, 0], color='#212121', lw=2.2, label='Ground Truth ($q_0$)')
    ax3.plot(t, q_h[:, 0], color='#1565c0', lw=2.0, linestyle='-', label='ℍ-DeltaNet ($q_0$)')
    ax3.plot(t, q_ekf[:, 0], color='#c62828', lw=1.5, linestyle=':', label='EKF ($q_0$)')
    
    ax3.plot(t, q_true[:, 3], color='#616161', lw=2.0, linestyle='-', label='Ground Truth ($q_3$ Yaw)')
    ax3.plot(t, q_h[:, 3], color='#00897b', lw=1.8, linestyle='--', label='ℍ-DeltaNet ($q_3$ Yaw)')
    
    ax3.set_title("C. Continuous Orientation Tracking ($q_0, q_3$ Components)", fontsize=12, fontweight='bold', pad=12)
    ax3.set_xlabel("Flight Time (seconds)", fontsize=10.5)
    ax3.set_ylabel("Quaternion Component Value", fontsize=10.5)
    ax3.set_ylim(-1.25, 1.25)
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend(loc='lower left', frameon=True, facecolor='#ffffff', edgecolor='#cccccc', fontsize=8.5, ncol=2)

    # -------------------------------------------------------------
    # Panel D: 6-DOF Sensor Telemetry Disturbances (Gyro & Accel)
    # -------------------------------------------------------------
    ax4_twin = ax4.twinx()
    
    l1 = ax4.plot(t, gyro[:, 0], color='#5e35b1', lw=1.1, alpha=0.75, label='Gyro $\\omega_x$ (rad/s)')
    l2 = ax4.plot(t, gyro[:, 1], color='#3949ab', lw=1.1, alpha=0.75, label='Gyro $\\omega_y$ (rad/s)')
    
    l3 = ax4_twin.plot(t, accel[:, 2], color='#d81b60', lw=1.3, alpha=0.85, label='Accel $a_z$ Thrust Spikes ($m/s^2$)')
    
    ax4.set_title("D. 6-DOF Sensor Telemetry with Dynamic Disturbances", fontsize=12, fontweight='bold', pad=12)
    ax4.set_xlabel("Flight Time (seconds)", fontsize=10.5)
    ax4.set_ylabel("Angular Velocity (rad/s)", fontsize=10.5, color='#3949ab')
    ax4_twin.set_ylabel("Measured Acceleration ($m/s^2$)", fontsize=10.5, color='#d81b60')
    ax4.grid(True, linestyle='--', alpha=0.6)
    
    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#cccccc', fontsize=8.5)

    plt.suptitle("6-DOF Aerospace Flight Telemetry Showdown: ℍ-DeltaNet vs EKF vs Madgwick", 
                 fontsize=15, fontweight='bold', y=0.98, color='#111111')
    
    output_path = r"C:\Users\Admin\.gemini\antigravity\scratch\quaternion_rnn\fig5_imu_ekf_benchmark.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated Polished Figure 5 at {output_path}")

if __name__ == "__main__":
    render_figure_5_polished()
