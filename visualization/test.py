import torch
import matplotlib.pyplot as plt
import numpy as np

def test_model_accuracy(device , model, h_val=1.0):
    model.eval()
    x_test = torch.linspace(-2, 2, 400).view(-1, 1).to(device)
    t_test = torch.linspace(0, 1, 100).view(-1, 1).to(device)
    
    probs_at_t = []
    times = [0.0, 0.5, 1.0]
    
    print("--- Numerical Accuracy Report ---")
    for t_val in times:
        t_t = (torch.ones_like(x_test) * t_val).to(device)
        h_t = (torch.ones_like(x_test) * h_val).to(device)
        with torch.no_grad():
            psi = model(x_test, t_t, h_t)
            prob = (psi[:, 0]**2 + psi[:, 1]**2).cpu().numpy()
            total_prob = np.trapz(prob, x_test.cpu().numpy().flatten())
            probs_at_t.append(total_prob)
            print(f"Total Probability at t={t_val}: {total_prob:.6f}")

    drift = np.abs(probs_at_t[-1] - probs_at_t[0]) / probs_at_t[0] * 100
    print(f"Probability Drift (Error): {drift:.2f}%")
    
    plt.figure(figsize=(10, 5))
    for i, t_val in enumerate(times):
        t_t = (torch.ones_like(x_test) * t_val).to(device)
        with torch.no_grad():
            psi = model(x_test, t_t, (torch.ones_like(x_test)*h_val).to(device))
            plt.plot(x_test.cpu().numpy(), (psi[:,0]**2 + psi[:,1]**2).cpu().numpy(), label=f't={t_val}')
    
    plt.title("Wave Profile Stability Over Time")
    plt.legend()
    plt.show()