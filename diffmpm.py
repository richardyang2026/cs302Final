import taichi as ti
import argparse
import os
import math
import numpy as np
import matplotlib.pyplot as plt

# Lab #4: Actuation and Control: From Open-Loop Patterns to Feedback & RL
#
# This implementation extends the previous DiffTaichi MPM simulation with actuation control.
# Open-loop control is implemented via a sinusoidal actuation pattern (in compute_actuation),
# and closed-loop control is achieved by gradient-based optimization of the actuation parameters
# (weights and bias) to maximize forward displacement (loss defined as negative average x-position).
#
# Use the command-line flag --control_mode with either "open_loop" (to run with fixed actuation)
# or "closed_loop" (to update control parameters via gradient descent). The default is "closed_loop".

# -------------------- INITIALIZATION & GLOBAL PARAMETERS --------------------
real = ti.f32
ti.init(default_fp=real, arch=ti.gpu, flatten_if=True)  # Using GPU backend

dim = 2
n_particles = 8192  # Will be updated by scene.finalize()
n_solid_particles = 0
n_actuators = 0
n_grid = 128         # Grid resolution for the MPM grid
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 1e-3
p_vol = 1
E = 20
mu = E
la = E
max_steps = 2048    # Maximum simulation steps (time history)
steps = 1024        # Default number of simulation steps used in forward simulation
gravity = 3.8
target = [0.8, 0.2]

# Deformation clamp parameters for polar decomposition:
F_min = 0.7
F_max = 1.3

# Define field types using lambda functions
scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(dim, dtype=real)
mat = lambda: ti.Matrix.field(dim, dim, dtype=real)

# -------------------- DECLARE FIELDS --------------------
actuator_id = ti.field(ti.i32)
particle_type = ti.field(ti.i32)
x, v = vec(), vec()
grid_v_in, grid_m_in = vec(), scalar()
grid_v_out = vec()
C, F = mat(), mat()

loss = scalar()

n_sin_waves = 4
weights = scalar()
bias = scalar()
x_avg = vec()

actuation = scalar()
actuation_omega = 20
act_strength = 4

# -------------------- ALLOCATION OF FIELDS --------------------
def allocate_fields():
    # Allocate Taichi fields for simulation data and gradients.
    ti.root.dense(ti.ij, (n_actuators, n_sin_waves)).place(weights)
    ti.root.dense(ti.i, n_actuators).place(bias)
    ti.root.dense(ti.ij, (max_steps, n_actuators)).place(actuation)
    ti.root.dense(ti.i, n_particles).place(actuator_id, particle_type)
    ti.root.dense(ti.k, max_steps).dense(ti.l, n_particles).place(x, v, C, F)
    ti.root.dense(ti.ij, n_grid).place(grid_v_in, grid_m_in, grid_v_out)
    ti.root.place(loss, x_avg)
    ti.root.lazy_grad()

# -------------------- CLEAR FIELDS KERNELS --------------------
@ti.kernel
def clear_grid():
    for i, j in grid_m_in:
        grid_v_in[i, j] = [0, 0]
        grid_m_in[i, j] = 0
        grid_v_in.grad[i, j] = [0, 0]
        grid_m_in.grad[i, j] = 0
        grid_v_out.grad[i, j] = [0, 0]

@ti.kernel
def clear_particle_grad():
    for f, i in x:
        x.grad[f, i] = [0, 0]
        v.grad[f, i] = [0, 0]
        C.grad[f, i] = [[0, 0], [0, 0]]
        F.grad[f, i] = [[0, 0], [0, 0]]

@ti.kernel
def clear_actuation_grad():
    for t, i in actuation:
        actuation[t, i] = 0.0

# -------------------- p2g KERNEL WITH DEFORMATION CLAMP --------------------
@ti.kernel
def p2g(f: ti.i32):
    # Transfer particle quantities to the grid.
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]

        # Compute new deformation gradient:
        new_F = (ti.Matrix.diag(dim=2, val=1) + dt * C[f, p]) @ F[f, p]

        # Polar decomposition + clamp to fix NaNs:
        r, s = ti.polar_decompose(new_F)
        # Clamp each singular value to [F_min, F_max]
        for i in ti.static(range(dim)):
            s[i, i] = ti.max(F_min, ti.min(s[i, i], F_max))
        new_F = r @ s

        # For fluid-like particles, force the shape to remain isotropic.
        J = new_F.determinant()
        if particle_type[p] == 0:  # fluid
            sqrtJ = ti.sqrt(J)
            new_F = ti.Matrix([[sqrtJ, 0], [0, sqrtJ]])

        F[f + 1, p] = new_F
        r, s = ti.polar_decompose(new_F)

        act_id = actuator_id[p]
        act = actuation[f, ti.max(0, act_id)] * act_strength
        if act_id == -1:
            act = 0.0
        A = ti.Matrix([[0, 0], [0, 1]]) * act

        cauchy = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        mass = 0.0
        if particle_type[p] == 0:
            mass = 4
            cauchy = ti.Matrix([[1.0, 0], [0, 0.1]]) * (J - 1) * E
        else:
            mass = 1
            cauchy = 2 * mu * (new_F - r) @ new_F.transpose() \
                     + ti.Matrix.diag(2, la * (J - 1) * J)

        cauchy += new_F @ A @ new_F.transpose()
        stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
        affine = stress + mass * C[f, p]

        # Transfer momentum to grid with boundary checks:
        base_i, base_j = base[0], base[1]
        for i_ in ti.static(range(3)):
            for j_ in ti.static(range(3)):
                pos = ti.Vector([base_i + i_, base_j + j_])
                if 0 <= pos[0] < n_grid and 0 <= pos[1] < n_grid:
                    dpos = (ti.cast(ti.Vector([i_, j_]), real) - fx) * dx
                    weight = w[i_][0] * w[j_][1]
                    grid_v_in[pos] += weight * (mass * v[f, p] + affine @ dpos)
                    grid_m_in[pos] += weight * mass

# -------------------- GRID OPERATION KERNEL --------------------
bound = 3
coeff = 0.5

@ti.kernel
def grid_op():
    for i, j in grid_m_in:
        inv_m = 1 / (grid_m_in[i, j] + 1e-10)
        v_out = inv_m * grid_v_in[i, j]
        v_out[1] -= dt * gravity
        if i < bound and v_out[0] < 0:
            v_out[0] = 0
            v_out[1] = 0
        if i > n_grid - bound and v_out[0] > 0:
            v_out[0] = 0
            v_out[1] = 0
        if j < bound and v_out[1] < 0:
            v_out[0] = 0
            v_out[1] = 0
            normal = ti.Vector([0.0, 1.0])
            lsq = (normal**2).sum()
            if lsq > 0.5:
                if ti.static(coeff < 0):
                    v_out[0] = 0
                    v_out[1] = 0
                else:
                    lin = v_out.dot(normal)
                    if lin < 0:
                        vit = v_out - lin * normal
                        lit = vit.norm() + 1e-10
                        if lit + coeff * lin <= 0:
                            v_out[0] = 0
                            v_out[1] = 0
                        else:
                            v_out = (1 + coeff * lin / lit) * vit
        if j > n_grid - bound and v_out[1] > 0:
            v_out[0] = 0
            v_out[1] = 0
        grid_v_out[i, j] = v_out

# -------------------- GRID-TO-PARTICLE (g2p) KERNEL --------------------
@ti.kernel
def g2p(f: ti.i32):
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, real)
        w = [0.5 * (1.5 - fx)**2,
             0.75 - (fx - 1.0)**2,
             0.5 * (fx - 0.5)**2]
        new_v = ti.Vector([0.0, 0.0])
        new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        for i_ in ti.static(range(3)):
            for j_ in ti.static(range(3)):
                pos = ti.Vector([base[0] + i_, base[1] + j_])
                if 0 <= pos[0] < n_grid and 0 <= pos[1] < n_grid:
                    dpos = ti.cast(ti.Vector([i_, j_]), real) - fx
                    weight = w[i_][0] * w[j_][1]
                    new_v += weight * grid_v_out[pos]
                    new_C += 4 * weight * grid_v_out[pos].outer_product(dpos) * inv_dx
        v[f + 1, p] = new_v
        x[f + 1, p] = x[f, p] + dt * new_v
        C[f + 1, p] = new_C

# -------------------- ACTUATION & LOSS --------------------
@ti.kernel
def compute_actuation(t: ti.i32):
    for i in range(n_actuators):
        act = 0.0
        for j in ti.static(range(n_sin_waves)):
            act += weights[i, j] * ti.sin(
                actuation_omega * t * dt + 2 * math.pi / n_sin_waves * j
            )
        act += bias[i]
        actuation[t, i] = ti.tanh(act)

# Modified compute_x_avg to accept the final time step as parameter.
@ti.kernel
def compute_x_avg(s_final: ti.i32):
    for i in range(n_particles):
        contrib = 0.0
        if particle_type[i] == 1:
            contrib = 1.0 / n_solid_particles
        ti.atomic_add(x_avg[None], contrib * x[s_final, i])

@ti.kernel
def compute_loss():
    # The loss is defined as the negative of the average x position of solid particles,
    # thereby incentivizing forward movement.
    loss[None] = -x_avg[None][0]

# -------------------- ADVANCE WITH AUTODIFF --------------------
@ti.ad.grad_replaced
def advance(s):
    clear_grid()
    compute_actuation(s)
    p2g(s)
    grid_op()
    g2p(s)

@ti.ad.grad_for(advance)
def advance_grad(s):
    clear_grid()
    p2g(s)
    grid_op()
    g2p.grad(s)
    grid_op.grad()
    p2g.grad(s)
    compute_actuation.grad(s)

def forward(total_steps=steps):
    for s in range(total_steps - 1):
        advance(s)
    x_avg[None] = [0, 0]
    compute_x_avg(total_steps - 2)
    compute_loss()

# -------------------- EXTENDED SCENE CLASS WITH TOPOLOGY --------------------
class Scene:
    def __init__(self):
        self.n_particles = 0
        self.n_solid_particles = 0
        self.x = []
        self.actuator_id = []
        self.particle_type = []
        self.offset_x = 0
        self.offset_y = 0
        self.parts = []
        self.connections = []

    def add_rect(self, x, y, w, h, actuation, ptype=1):
        # Use a factor of 1.5 (instead of 2) to slightly reduce the particle count.
        factor = 1.5
        base_w = int(w / dx)
        base_h = int(h / dx)
        w_count = int(base_w * factor)
        h_count = int(base_h * factor)

        if w_count <= 0:
            w_count = 1
        if h_count <= 0:
            h_count = 1

        real_dx = w / w_count
        real_dy = h / h_count
        if ptype == 0:
            assert actuation == -1

        for i in range(w_count):
            for j in range(h_count):
                self.x.append([
                    x + (i + 0.5) * real_dx + self.offset_x,
                    y + (j + 0.5) * real_dy + self.offset_y
                ])
                self.actuator_id.append(actuation)
                self.particle_type.append(ptype)
                self.n_particles += 1
                self.n_solid_particles += int(ptype == 1)

        # Record the rectangle as a geometric part (for topology).
        part_center = [x + w / 2 + self.offset_x, y + h / 2 + self.offset_y]
        self.parts.append({
            'center': part_center,
            'width': w,
            'height': h,
            'actuation': actuation,
            'ptype': ptype
        })

    def add_connection(self, part_idx1, part_idx2):
        self.connections.append((part_idx1, part_idx2))

    def set_offset(self, x, y):
        self.offset_x = x
        self.offset_y = y

    def finalize(self):
        global n_particles, n_solid_particles
        n_particles = self.n_particles
        n_solid_particles = self.n_solid_particles
        print('n_particles', n_particles)
        print('n_solid', n_solid_particles)

    def set_n_actuators(self, n_act):
        global n_actuators
        n_actuators = n_act

# -------------------- BUNNY GENERATOR --------------------
def procedural_bunny(scene):
    # Bunny Body
    body_x = 0.2
    body_y = 0.3
    body_w = 0.3
    body_h = 0.15
    scene.add_rect(body_x, body_y, body_w, body_h, -1, ptype=1)
    body_index = len(scene.parts) - 1

    # Bunny Head
    head_x = body_x + body_w
    head_y = body_y + body_h * 0.5 - 0.075
    head_w = 0.15
    head_h = 0.15
    scene.add_rect(head_x, head_y, head_w, head_h, -1, ptype=1)
    head_index = len(scene.parts) - 1
    scene.add_connection(body_index, head_index)

    # Bunny Ears
    ear1_x = head_x + head_w * 0.3 - 0.015
    ear1_y = head_y + head_h
    ear1_w = 0.03
    ear1_h = 0.1
    scene.add_rect(ear1_x, ear1_y, ear1_w, ear1_h, -1, ptype=1)
    ear1_index = len(scene.parts) - 1
    scene.add_connection(head_index, ear1_index)
   
    ear2_x = head_x + head_w * 0.7 - 0.015
    ear2_y = head_y + head_h
    ear2_w = 0.03
    ear2_h = 0.1
    scene.add_rect(ear2_x, ear2_y, ear2_w, ear2_h, -1, ptype=1)
    ear2_index = len(scene.parts) - 1
    scene.add_connection(head_index, ear2_index)

    # Bunny Tail
    tail_x = body_x - 0.05
    tail_y = body_y + body_h * 0.5 - 0.025
    tail_w = 0.05
    tail_h = 0.05
    scene.add_rect(tail_x, tail_y, tail_w, tail_h, -1, ptype=1)
    tail_index = len(scene.parts) - 1
    scene.add_connection(body_index, tail_index)

    # Bunny Front Paw
    paw_x = head_x + head_w * 0.5 - 0.025
    paw_y = head_y - 0.1
    paw_w = 0.05
    paw_h = 0.1
    scene.add_rect(paw_x, paw_y, paw_w, paw_h, 0, ptype=1)
    paw_index = len(scene.parts) - 1
    scene.add_connection(head_index, paw_index)

    # Bunny Hind Legs
    leg1_x = body_x + body_w * 0.2 - 0.025
    leg1_y = body_y - 0.1
    leg1_w = 0.05
    leg1_h = 0.1
    scene.add_rect(leg1_x, leg1_y, leg1_w, leg1_h, 1, ptype=1)
    leg1_index = len(scene.parts) - 1
    scene.add_connection(body_index, leg1_index)
   
    leg2_x = body_x + body_w * 0.8 - 0.025
    leg2_y = body_y - 0.1
    leg2_w = 0.05
    leg2_h = 0.1
    scene.add_rect(leg2_x, leg2_y, leg2_w, leg2_h, 2, ptype=1)
    leg2_index = len(scene.parts) - 1
    scene.add_connection(body_index, leg2_index)

    scene.set_n_actuators(3)

# -------------------- OLD ROBOT GENERATOR --------------------
def robot(scene):
    # This function is the old robot generator, kept for reference.
    scene.set_offset(0.1, 0.03)
    scene.add_rect(0.0, 0.1, 0.3, 0.1, -1)
    scene.add_rect(0.0, 0.0, 0.05, 0.1, 0)
    scene.add_rect(0.05, 0.0, 0.05, 0.1, 1)
    scene.add_rect(0.2, 0.0, 0.05, 0.1, 2)
    scene.add_rect(0.25, 0.0, 0.05, 0.1, 3)
    scene.add_rect(-0.05, 0.12, 0.05, 0.04, -1)
    scene.add_rect(0.225, 0.2, 0.075, 0.1, -1)
    scene.add_rect(0.225, 0.3, 0.15, 0.07, -1)
    scene.set_n_actuators(4)

# -------------------- TOPOLOGY VISUALIZATION --------------------
def visualize_topology(scene, folder):
    fig, ax = plt.subplots()
    for part in scene.parts:
        center = part['center']
        ax.plot(center[0], center[1], 'bo')
        rect = plt.Rectangle(
            (center[0]-part['width']/2, center[1]-part['height']/2),
            part['width'], part['height'], fill=False, edgecolor='b'
        )
        ax.add_patch(rect)
    for conn in scene.connections:
        part1 = scene.parts[conn[0]]
        part2 = scene.parts[conn[1]]
        c1 = part1['center']
        c2 = part2['center']
        ax.plot([c1[0], c2[0]], [c1[1], c2[1]], 'r--')
    plt.title("Topology of Procedural Robot/Bunny")
    plt.xlabel("X")
    plt.ylabel("Y")
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, "topology.png"))
    plt.close()

# -------------------- GUI VISUALIZATION --------------------
gui = ti.GUI("Differentiable MPM", (640, 640), background_color=0xFFFFFF)

def visualize(s, folder):
    aid = actuator_id.to_numpy()
    colors = np.empty(shape=n_particles, dtype=np.uint32)
    particles = x.to_numpy()[s]
    actuation_ = actuation.to_numpy()
    for i in range(n_particles):
        color = 0x111111
        if aid[i] != -1:
            a_ = actuation_[s - 1, int(aid[i])]
            color = ti.rgb_to_hex((0.5 - a_, 0.5 - abs(a_), 0.5 + a_))
        colors[i] = color
    gui.circles(pos=particles, color=colors, radius=1.5)
    gui.line((0.05, 0.02), (0.95, 0.02), radius=3, color=0x0)
    os.makedirs(folder, exist_ok=True)
    gui.show(f'{folder}/{s:04d}.png')

# -------------------- MAIN FUNCTION --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=100, help='Number of optimization iterations (for closed-loop control)')
    parser.add_argument('--control_mode', type=str, default='closed_loop', choices=['open_loop', 'closed_loop'],
                        help='Control mode: open_loop (fixed actuation) or closed_loop (gradient-based control)')
    options = parser.parse_args()

    # Build the bunny structure.
    scene = Scene()
    procedural_bunny(scene)
    scene.finalize()
    allocate_fields()

    # Initialize weights randomly.
    for i in range(n_actuators):
        for j in range(n_sin_waves):
            weights[i, j] = np.random.randn() * 0.01

    # Initialize particle fields from scene data.
    for i in range(scene.n_particles):
        x[0, i] = scene.x[i]
        F[0, i] = [[1, 0], [0, 1]]
        actuator_id[i] = scene.actuator_id[i]
        particle_type[i] = scene.particle_type[i]

    if options.control_mode == 'closed_loop':
        # Optimization loop for closed-loop control: gradient descent updates on actuation parameters.
        losses = []
        for iter in range(options.iters):
            with ti.ad.Tape(loss):
                forward()
            l = loss[None]
            losses.append(l)
            print('Iteration:', iter, 'loss:', l)

            # Simple gradient descent update for actuation parameters.
            lr = 0.1
            for i in range(n_actuators):
                for j in range(n_sin_waves):
                    weights[i, j] -= lr * weights.grad[i, j]
                bias[i] -= lr * bias.grad[i]

            # Every 10 iterations, run a longer forward pass for visualization.
            if iter % 10 == 0:
                forward(1500)
                for s in range(15, 1500, 16):
                    visualize(s, f'diffmpm/iter{iter:03d}/')

        # Plot the loss curve.
        plt.title("Optimization of Forward Displacement")
        plt.ylabel("Loss (Negative x-Average)")
        plt.xlabel("Iterations")
        plt.plot(losses)
        plt.show()

        # Visualize the connectivity (topology).
        visualize_topology(scene, 'diffmpm/topology/')
    else:
        # Open-loop control: Run simulation without updating actuation parameters.
        forward(1500)
        for s in range(15, 1500, 16):
            visualize(s, 'diffmpm/open_loop/')
        print("Open-loop simulation loss (negative x-average):", loss[None])

if __name__ == '__main__':
    main()
