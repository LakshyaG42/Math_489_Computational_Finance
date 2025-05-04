import numpy as np
import plotly.graph_objects as go
import plotly.express as px
T = 1.0
N = 100
dt = T / N
t = np.linspace(0, T, N+1)
np.random.seed(276786726)


# 1.1 Simulate Brownian increments and path:
increments = np.sqrt(dt) * np.random.randn(N)
B = np.concatenate(([0], np.cumsum(increments)))

fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=B, mode='lines', name='B(t)'))
fig.update_layout(
    title='Simulated Brownian Motion Path on [0,1]',
    xaxis_title='Time t',
    yaxis_title='B(t) value'
)
fig.show()

# 1.2 Simulate the Ornstein-Uhlenbeck process using both explicit and implicit Euler methods. 
Y_explicit = np.zeros(N+1)
Y_implicit = np.zeros(N+1)

for i in range(N):
    dB = B[i+1] - B[i]  
    # Explicit Euler:
    Y_explicit[i+1] = Y_explicit[i] - Y_explicit[i]*dt + dB
    # Implicit Euler:
    Y_implicit[i+1] = (Y_implicit[i] + dB) / (1 + dt)
fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=Y_explicit, mode='lines', name='Explicit Euler'))
fig.add_trace(go.Scatter(x=t, y=Y_implicit, mode='lines', name='Implicit Euler'))
fig.update_layout(
    title='Ornstein–Uhlenbeck Process Simulation',
    xaxis_title='Time t',
    yaxis_title='Y(t)'
)
fig.show()


# 1.3 Simulate the Feller process using both explicit and Milstein schemes.
Z_explicit = np.zeros(N+1)
Z_milstein = np.zeros(N+1)
Z_explicit[0] = 0
Z_milstein[0] = 0

for i in range(N):
    dB = B[i+1] - B[i]
    # Explicit Euler scheme:
    drift_exp = (0.5 * np.exp(-Z_explicit[i]) - 1) * dt
    diffusion_exp = np.exp(-Z_explicit[i] / 2) * dB
    Z_explicit[i+1] = Z_explicit[i] + drift_exp + diffusion_exp

    # Milstein scheme:
    drift_mil = (0.5 * np.exp(-Z_milstein[i]) - 1) * dt
    diffusion_mil = np.exp(-Z_milstein[i] / 2) * dB
    correction = - (1/(4*np.exp(Z_milstein[i]))) * ((dB)**2 - dt)
    Z_milstein[i+1] = Z_milstein[i] + drift_mil + diffusion_mil + correction

Y_explicit_from_Z = np.exp(Z_explicit)
Y_milstein_from_Z  = np.exp(Z_milstein)

fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=Y_explicit_from_Z, mode='lines', name='Euler for Z -> Y'))
fig.add_trace(go.Scatter(x=t, y=Y_milstein_from_Z, mode='lines', name='Milstein for Z -> Y'))
fig.update_layout(
    title='Feller Process Simulated via Log Transformation',
    xaxis_title='Time t',
    yaxis_title='Y(t)'
)
fig.show()

# 1.4 Simulate the Geometric Brownian motion process using the log transformation method.
Z_gbm = np.zeros(N+1)
Z_gbm[0] = 0

for i in range(N):
    dB = B[i+1] - B[i]
    Z_gbm[i+1] = Z_gbm[i] + 0.5*dt + dB

Y_gbm = np.exp(Z_gbm)

fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=Y_gbm, mode='lines', name='Geometric Brownian Motion'))
fig.update_layout(
    title='Simulation of Geometric Brownian Motion via Log Transformation',
    xaxis_title='Time t',
    yaxis_title='Y(t)'
)
fig.show()

# 1.5 Simulate the Brownian bridge process using both explicit and implicit Euler methods.
Y_bb_explicit = np.zeros(N+1)
Y_bb_implicit = np.zeros(N+1)

for i in range(N):
    dB = B[i+1] - B[i]
    # Explicit Euler update:
    Y_bb_explicit[i+1] = Y_bb_explicit[i] - (Y_bb_explicit[i] / (1 - t[i])) * dt + dB
    # Implicit Euler update:
    Y_bb_implicit[i+1] = (Y_bb_implicit[i] + dB) / (1 + dt/(1 - t[i+1]))

fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=Y_bb_explicit, mode='lines', name='Explicit Euler (Bridge)'))
fig.add_trace(go.Scatter(x=t, y=Y_bb_implicit, mode='lines', name='Implicit Euler (Bridge)'))
fig.update_layout(
    title='Brownian Bridge Process Simulation',
    xaxis_title='Time t',
    yaxis_title='Y(t)'
)
fig.show()

# 1.6 Simulate the process dY = Y B dB using the explicit Euler method.
Y_mult = np.zeros(N+1)
Y_mult[0] = 1
for i in range(N):
    dB = B[i+1] - B[i]
    Y_mult[i+1] = Y_mult[i] + Y_mult[i] * B[i] * dB

# To check if positive:
print("Minimum value of Y_mult:", Y_mult.min())

fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=Y_mult, mode='lines', name='Process dY = Y B dB'))
fig.update_layout(
    title='Process with Multiplicative Noise',
    xaxis_title='Time t',
    yaxis_title='Y(t)'
)
fig.show()
