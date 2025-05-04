import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, gamma, norm
def sample_positive_stable(alpha, size=1):
    # U ~ Uniform(0, pi) and W ~ Exponential(1)
    U = np.random.uniform(0, np.pi, size)
    W = expon.rvs(scale=1, size=size)

    # Y = (sin(alpha * U) / (sin(U))**(1/alpha)) * (sin((1 - alpha) * U) / W)**((1 - alpha) / alpha)
    factor1 = np.sin(alpha * U) / (np.sin(U))**(1/alpha)
    factor2 = (np.sin((1 - alpha) * U) / W)**((1 - alpha) / alpha)
    Y = factor1 * factor2
    return Y

def generate_gumbel_copula(theta, n=1000):
    """
    Generate n samples (U1, U2) from the Gumbel copula with parameter theta.
    """
    alpha = 1.0/theta
    Y = sample_positive_stable(alpha, size=n)
    # Draw n independent exponential random variables for each U
    E1 = expon.rvs(scale=1, size=n)
    E2 = expon.rvs(scale=1, size=n)
    U1 = np.exp(- (E1 / Y)**(1.0/theta))
    U2 = np.exp(- (E2 / Y)**(1.0/theta))
    return U1, U2
# Plot Gumbel copula samples for theta = 2, 5, 50
thetas = [2, 5, 50]
plt.figure(figsize=(15,4))
for i, theta in enumerate(thetas, 1):
    U1, U2 = generate_gumbel_copula(theta, n=1000)
    plt.subplot(1, len(thetas), i)
    plt.scatter(U1, U2, s=10, alpha=0.6)
    plt.title(f'Gumbel Copula, theta={theta}')
    plt.xlabel('U1')
    plt.ylabel('U2')
plt.tight_layout()
plt.show()

### Part 4.2: Clayton Copula via Marshall-Olkin

def generate_clayton_copula(theta, n=1000):
    """
    Generate n samples (U1, U2) from the Clayton copula with parameter theta.
    """
    # Y ~ Gamma(shape=1/theta, scale=1)
    shape = 1.0/theta
    Y = gamma.rvs(a=shape, scale=1, size=n)
    # Draw independent exponential random variables
    E1 = expon.rvs(scale=1, size=n)
    E2 = expon.rvs(scale=1, size=n)

    U1 = (1 + E1 / Y)**(-1.0/theta)
    U2 = (1 + E2 / Y)**(-1.0/theta)
    return U1, U2
# Plot Clayton copula samples for theta = 2, 5, 50
plt.figure(figsize=(15,4))
for i, theta in enumerate(thetas, 1):
    U1, U2 = generate_clayton_copula(theta, n=1000)
    plt.subplot(1, len(thetas), i)
    plt.scatter(U1, U2, s=10, alpha=0.6)
    plt.title(f'Clayton Copula, theta={theta}')
    plt.xlabel('U1')
    plt.ylabel('U2')
plt.tight_layout()
plt.show()

### Part 4.3: Gaussian Copula via Cholesky Factorization

def generate_gaussian_copula(rho, n=1000):
    """
    Generate n samples (U1, U2) from the Gaussian copula with correlation rho.
    """
    Z = np.random.randn(n, 2)

    cov = np.array([[1, rho],
                    [rho, 1]])
    L = np.linalg.cholesky(cov)
    X = Z @ L.T  
    # Map to uniforms using the standard normal CDF
    U = norm.cdf(X)
    return U[:,0], U[:,1]

rhos = [-0.9, -0.5, 0, 0.5, 0.9]
plt.figure(figsize=(15,8))
for i, rho in enumerate(rhos, 1):
    U1, U2 = generate_gaussian_copula(rho, n=1000)
    plt.subplot(3, 2, i)
    plt.scatter(U1, U2, s=10, alpha=0.6)
    plt.title(f'Gaussian Copula, rho={rho}')
    plt.xlabel('U1')
    plt.ylabel('U2')
plt.tight_layout()
plt.show()
