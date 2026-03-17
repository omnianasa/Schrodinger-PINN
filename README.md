# Quantum Tunneling PINN: Solving the Schrödinger Equation with a Double-Well Potential

This project uses a **Physics-Informed Neural Network (PINN)** to solve the time dependent Schrödinger equation for a particle in a double‑well potential. The potential barrier height $H$ is treated as an additional input, allowing the model to learn the wave function dynamics for a continuous range of $H$.

## Overview

- **Equation**:  
  $$i\frac{\partial \psi}{\partial t} = -\frac{1}{2}\frac{\partial^2 \psi}{\partial x^2} + V(x; H)\psi$$
  where $$V(x; H) = H(x^4 - 2x^2 + 1.0)$$ is a symmetric double-well potential.
- **Boundary conditions**: $$\psi(\pm 2, t) = 0$$
- **Initial condition**: Gaussian wave packet centered at $x = -1$ with momentum $k_0 = 5$:
  $$\psi(x,0) = e^{-\frac{(x+1)^2}{2\sigma^2}} e^{i k_0 x}$$
- **Domain**: $x \in [-2, 2]$, $t \in [0, 1]$, $H \in [1, 3]$ (training range).

## Model Architecture

- **Input**: $(x, t, H)$
- **Output**: $(\text{Re}(\psi), \text{Im}(\psi))$
- **Network**: 3 → 256 → 256 → 256 → 256 → 2 with **Sine activation** and **Residual connections**.
- **Loss function** combines:
  - PDE residuals (Schrödinger equation)
  - Initial condition (IC)
  - Boundary condition (BC)
  - Normalization (probability conservation)

## Training

Two‑stage training process:
1. **Adam** optimizer (8000 epochs) with a `ReduceLROnPlateau` scheduler.
2. **L‑BFGS** fine‑tuning (1500 iterations) for high-precision convergence.

## Results Summary

The model's accuracy is measured by the **Probability Drift**, defined as:
$$\text{Drift} = \frac{|\int |\psi(x, t=1)|^2 dx - \int |\psi(x, t=0)|^2 dx|}{\int |\psi(x, t=0)|^2 dx} \times 100\%$$

| $H$ | Drift (%) |
| :--- | :--- | 
| **0.5** | 5.96 | 
| **1.0** | 2.61 | 
| **2.0** | 1.78 | 
| **3.0** | 4.15 |
| **4.0** | 35.33 |
| **5.0** | 53.45 | 

---

## Detailed Visualizations per $H$

### Comparative Results per Barrier Height ($H$)

---

### $H = 0.5$ 

| Heatmap | 3D Surface |
| :---: | :---: |
| <img src="results/heat0.5.png" width="400" height="300"> | <img src="results/w3d0.5.png" width="400" height="300"> |

| Stability Check | Wave Snapshots |
| :---: | :---: |
| <img src="results/stab0.5.png" width="400" height="300"> | <img src="results/snap0.5.png" width="400" height="300"> |

---

### $H = 1.0$ 

| Heatmap | 3D Surface |
| :---: | :---: |
| <img src="results/heat1.png" width="400" height="300"> | <img src="results/w3d1.png" width="400" height="300"> |

| Stability Check | Wave Snapshots |
| :---: | :---: |
| <img src="results/stab1.png" width="400" height="300"> | <img src="results/snap1.png" width="400" height="300"> |

---

### $H = 2.0$ 

| Heatmap | 3D Surface |
| :---: | :---: |
| <img src="results/heat2.png" width="400" height="300"> | <img src="results/w3d2.png" width="400" height="300"> |

| Stability Check | Wave Snapshots |
| :---: | :---: |
| <img src="results/stab2.png" width="400" height="300"> | <img src="results/snap2.png" width="400" height="300"> |

---

### $H = 3.0$ 

| Heatmap | 3D Surface |
| :---: | :---: |
| <img src="results/heat3.png" width="400" height="300"> | <img src="results/w3d3.png" width="400" height="300"> |

| Stability Check | Wave Snapshots |
| :---: | :---: |
| <img src="results/stab3.png" width="400" height="300"> | <img src="results/snap3.png" width="400" height="300"> |

---

### $H = 4.0$ 

| Heatmap | 3D Surface |
| :---: | :---: |
| <img src="results/heat4.png" width="400" height="300"> | <img src="results/w3d4.png" width="400" height="300"> |

| Stability Check | Wave Snapshots |
| :---: | :---: |
| <img src="results/stab4.png" width="400" height="300"> | <img src="results/snap4.png" width="400" height="300"> |

---

### $H = 5.0$ 

| Heatmap | 3D Surface |
| :---: | :---: |
| <img src="results/heat5.png" width="400" height="300"> | <img src="results/w3d5.png" width="400" height="300"> |

| Stability Check | Wave Snapshots |
| :---: | :---: |
| <img src="results/stab5.png" width="400" height="300"> | <img src="results/snap5.png" width="400" height="300"> |

## Key Observations

* **Stability**: For $H \leq 3.0$, the model maintains high physical fidelity. The use of the $Sin$ activation function allows the network to capture the oscillatory nature of the complex wave function.
* **Extrapolation**: Performance degrades sharply for $H > 3.0$. This is expected as the PINN has not seen the steeper potential gradients during training.
* **Conservation**: The probability normalization loss term was critical in preventing the wave function from vanishing or exploding over time.