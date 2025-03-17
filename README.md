# DiffMPM: Differentiable Physics for Motion Optimization  

## Overview  
This project explores **differentiable physics simulation** using the **Material Point Method (MPM)** to model soft-body motion. It integrates **gradient-based optimization** to improve actuation control, allowing a **procedurally generated bunny** to learn forward movement.  

- **Taichi-powered GPU acceleration** for high-performance simulations  
- **Open-loop vs Closed-loop control** comparison  
- **Automatic differentiation** for optimizing movement via **gradient descent**  

##  Key Features  
### ** Procedural Structure Generation**  
The bunny is represented as a collection of **rectangular segments** connected to form a soft-body system. A **Scene class** records both the **geometry** (positions, dimensions) and **topology** (connections between parts).  

### ** Physics Simulation using MPM**  
The simulation follows a **Particle-In-Cell (PIC) approach** with:  
- **Particle-to-Grid (P2G) transfer:** Particles deposit momentum onto a computational grid.  
- **Grid Update:** Forces such as **gravity** and **actuation** update velocities.  
- **Grid-to-Particle (G2P) transfer:** The updated velocities propagate back to the particles.  

### ** Control Strategies**  
- **Open-loop control:** Uses a predefined sinusoidal actuation pattern.  
- **Closed-loop control:** Optimizes actuation parameters using **gradient-based learning** to maximize forward displacement.  

## Installation  
### ** Prerequisites**
- Python 3.8+  
- Taichi (`pip install taichi`)  
- NumPy, Matplotlib (`pip install numpy matplotlib`)  

### Running the Code 
1. Clone the repository:  
   ```bash
   git clone https://github.com/richardyang2026/cs302Final.git
   cd DiffMPM
