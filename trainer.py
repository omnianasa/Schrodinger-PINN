import torch
import torch.nn as nn

from pinn.model_arch import DoubleWallModel
from SchrodingerLoss.loss import compute_loss
from visualization.visualizer import Visualizer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DoubleWallModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=500, factor=0.5)

    print("Phase 1: Adam Training...")
    for epoch in range(8001):
        optimizer.zero_grad()
        pde_w = min(1.0, epoch / 4000.0)
        
        x_col = ((torch.rand(5000, 1) * 4) - 2).to(device)
        t_col = torch.rand(5000, 1).to(device)
        H_col = (torch.rand(5000, 1) * 2.0 + 1.0).to(device)
        x_init = ((torch.rand(1000, 1) * 4) - 2).to(device)
        
        loss = compute_loss(device, epoch, model, x_col, t_col, H_col, x_init, pde_w)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(loss)
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

    print("Phase 2: L-BFGS Fine-tuning...")
    optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=1500, history_size=100)

    def closure():
        optimizer_lbfgs.zero_grad()
        loss = compute_loss(model, x_col, t_col, H_col, x_init, 1.0)
        loss.backward()
        return loss

    optimizer_lbfgs.step(closure)
    print("Training Complete!\nLet's Start Visualization")

    visualizer = Visualizer(device)
    visualizer.plot_final_heatmap(model)
    visualizer.plot_final_heatmap(model, h_val=5)
    visualizer.plot_final_3d(model)
    visualizer.plot_stability_check(model)
    visualizer.plot_wave_snapshots(model)


if __name__ == "__main__":
    main()
