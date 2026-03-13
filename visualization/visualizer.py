import matplotlib.pyplot as plt
from utils.double_wall import get_potential
import torch
import numpy as np


class Visualizer:

    def __init__(self, device):
        self.device = device


    def plot_final_heatmap(self, model, h_val=1.0):
        x_np = np.linspace(-2, 2, 200)
        t_np = np.linspace(0, 1, 100)
        X, T = np.meshgrid(x_np, t_np)
        x_t = torch.tensor(X.flatten(), dtype=torch.float32).view(-1, 1).to(self.device)
        t_t = torch.tensor(T.flatten(), dtype=torch.float32).view(-1, 1).to(self.device)
        h_t = (torch.ones_like(x_t) * h_val).to(self.device)
        
        model.eval()
        with torch.no_grad():
            psi = model(x_t, t_t, h_t)
            prob = (psi[:,0]**2 + psi[:,1]**2).cpu().numpy().reshape(X.shape)
        
        plt.figure(figsize=(10, 8))
        plt.pcolormesh(X, T, prob, cmap='magma', shading='auto')
        plt.colorbar(label='Probability Density')
        plt.axvline(x=0, color='white', linestyle='--', alpha=0.5)
        plt.title(f"Quantum Tunneling Spacetime Heatmap (Fixed Norm) | H={h_val}")
        plt.xlabel("Position (x)")
        plt.ylabel("Time (t)")
        plt.show()

    def plot_final_3d(self, model, h_val=1.0):
        x_np = np.linspace(-2, 2, 100)
        t_np = np.linspace(0, 1, 100)
        X, T = np.meshgrid(x_np, t_np)
        
        x_t = torch.tensor(X.flatten(), dtype=torch.float32).view(-1, 1).to(self.device)
        t_t = torch.tensor(T.flatten(), dtype=torch.float32).view(-1, 1).to(self.device)
        h_t = (torch.ones_like(x_t) * h_val).to(self.device)
        
        model.eval()
        with torch.no_grad():
            psi = model(x_t, t_t, h_t)
            prob = (psi[:,0]**2 + psi[:,1]**2).cpu().numpy().reshape(X.shape)
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, T, prob, cmap='magma', edgecolor='none')
        
        ax.set_title(f"Quantum Wave Evolution | Barrier H={h_val}")
        ax.set_xlabel("Position (x)")
        ax.set_ylabel("Time (t)")
        ax.set_zlabel("Probability Density")
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        plt.show()


    def plot_stability_check(self, model, h_val=1.0):
        model.eval()
        x_test = torch.linspace(-2, 2, 500).view(-1, 1).to(self.device)

        with torch.no_grad():
            # t = 0
            psi0 = model(x_test, torch.zeros_like(x_test), torch.ones_like(x_test)*h_val)
            prob0 = (psi0[:,0]**2 + psi0[:,1]**2).cpu().numpy()
            
            # t = 1
            psi1 = model(x_test, torch.ones_like(x_test), torch.ones_like(x_test)*h_val)
            prob1 = (psi1[:,0]**2 + psi1[:,1]**2).cpu().numpy()

        plt.figure(figsize=(10, 6))
        plt.plot(x_test.cpu().numpy(), prob0, 'b-', label='Initial State (t=0)', linewidth=2)
        plt.plot(x_test.cpu().numpy(), prob1, 'r--', label='Final State (t=1)', linewidth=2)
        
        V = get_potential(x_test.cpu(), h_val).numpy()
        plt.fill_between(x_test.cpu().numpy().flatten(), 0, V.flatten()*0.1, color='gray', alpha=0.2, label='Potential Barrier (Scaled)')

        plt.title(f"Wavefunction Stability Analysis | Drift: 5.04%")
        plt.xlabel("Position (x)")
        plt.ylabel("Probability Density")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()


    def plot_wave_snapshots(self, model, h_val=1.0):
        times = [0.0, 0.25, 0.5, 0.75]
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
        axes = axes.flatten()
        
        x_np = np.linspace(-2, 2, 200)
        x_t = torch.tensor(x_np, dtype=torch.float32).view(-1, 1).to(self.device)
        h_t = (torch.ones_like(x_t) * h_val).to(self.device)
        
        for i, t_val in enumerate(times):
            t_t = (torch.ones_like(x_t) * t_val).to(self.device)
            model.eval()
            with torch.no_grad():
                psi = model(x_t, t_t, h_t)
                prob = (psi[:,0]**2 + psi[:,1]**2).cpu().numpy()
                
            axes[i].plot(x_np, prob, color='blue', lw=2)

            axes[i].axvline(x=0, color='red', linestyle='--', alpha=0.3)
            axes[i].set_title(f"Time t={t_val:.2f}")
            axes[i].grid(True, alpha=0.3)
            if i >= 2: axes[i].set_xlabel("Position (x)")
            if i % 2 == 0: axes[i].set_ylabel("Probability Density")
            
        plt.suptitle(f"Quantum Wave Snapshots | H={h_val}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()

