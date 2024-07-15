import numpy as np
import matplotlib.pyplot as plt

# Parameters
gamma_a = 1.0
r_a = 1.0
Q_a = 1.0
L = 1.0  # Length of the spatial domain
T = 1.0  # Total time
Nx = 50  # Number of spatial points
Nt = 50  # Number of time points

# Discretization: sheet model
dx = L / (Nx - 1)
dt = T / (Nt - 1)

x = np.linspace(0, L, Nx)
t = np.linspace(0, T, Nt)

# Initialize the solution array
phi = np.zeros((Nt, Nx))

# Initial and boundary conditions
phi[0, :, :] = 0  # Initial condition at t=0
phi[:, 0, :] = 0  # Boundary condition at x=0
phi[:, :, 0] = 0  # Boundary condition at y=0
phi[:, L, :] = 0  # Boundary condition at x=L
phi[:, :, L] = 0  # Boundary condition at x=L

# Finite difference coefficients
alpha = dt**2 / gamma_a**2
beta = 2 * dt / gamma_a
gamma = r_a**2 * dt**2 / dx**2

# Time-stepping loop
for n in range(1, Nt-1):
    for i in range(1, Nx-1):
        phi[n+1, i] = (alpha * (phi[n, i+1] - 2*phi[n, i] + phi[n, i-1])
                       + beta * (phi[n, i] - phi[n-1, i])
                       + 2 * phi[n, i] - phi[n-1, i]
                       + dt**2 * Q_a)

# Plot the results
X, T = np.meshgrid(x, t)
plt.contourf(X, T, phi, cmap='viridis')
plt.colorbar()
plt.xlabel('Space')
plt.ylabel('Time')
plt.title('Solution of the Differential Equation')
plt.show()
