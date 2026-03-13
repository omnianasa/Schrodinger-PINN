import torch

from utils.double_wall import get_potential

def compute_loss(device, epoch, model, x, t, H, x_init, pde_weight):
    x.requires_grad_(True)
    t.requires_grad_(True)

    psi = model(x, t, H)
    u, v = psi[:, 0:1], psi[:, 1:2]

    # Derivatives
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    v_t = torch.autograd.grad(v, t, torch.ones_like(v), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0]

    V = get_potential(x, H)
    f_u = -v_t + 0.5 * u_xx - V * u
    f_v =  u_t + 0.5 * v_xx - V * v
    pde_multiplier = 100.0 + (1900.0 * (epoch / 8000.0))
    loss_pde = torch.mean(f_u**2 + f_v**2) * pde_multiplier

    # Initial Condition (t=0)
    t0 = torch.zeros_like(x_init)
    psi0_pred = model(x_init, t0, H[:len(x_init)])
    x0, sigma, k0 = -1.0, 0.2, 5.0
    target_u0 = torch.exp(-(x_init-x0)**2/(2*sigma**2)) * torch.cos(k0*x_init)
    target_v0 = torch.exp(-(x_init-x0)**2/(2*sigma**2)) * torch.sin(k0*x_init)
    loss_ic = torch.mean((psi0_pred[:, 0:1]-target_u0)**2 + (psi0_pred[:, 1:2]-target_v0)**2)

    # Boundary Conditions (x = -2, 2)
    x_bc = torch.tensor([[-2.0], [2.0]], device=device).repeat(100, 1)
    t_bc = torch.rand(200, 1, device=device)
    psi_bc = model(x_bc, t_bc, H[:200])
    loss_bc = torch.mean(psi_bc**2)

    # Relative Normalization
    prob0 = torch.mean(psi0_pred[:,0]**2 + psi0_pred[:,1]**2)
    prob_t = torch.mean(u**2 + v**2)
    loss_norm = (prob_t - prob0)**2

    #total Loss with tuned weights
    return (100.0 * pde_weight * loss_pde + 
            10000.0 * loss_ic + 
            1000.0 * loss_bc + 
            5000.0 * loss_norm)