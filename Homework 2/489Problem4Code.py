import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

def generate_normals_box_muller(n):
    """
    Generate n samples from N(0,1) using the Box-Müller transform.
    """
    # We'll generate n//2 pairs of samples, so need n to be even will handle odd case later
    n2 = n // 2
    u1 = np.random.rand(n2)
    u2 = np.random.rand(n2)

    # Box-Müller formulas
    r = np.sqrt(-2.0 * np.log(u1))
    theta = 2.0 * np.pi * u2
    z0 = r * np.cos(theta)
    z1 = r * np.sin(theta)

    z = np.concatenate((z0, z1))

    # If n is odd, generate one extra
    if n % 2 == 1:
        z_extra = generate_normals_box_muller(1)
        z = np.concatenate((z, z_extra))

    return z[:n]


def generate_normals_accept_reject(n):
    """
    Generate n samples from N(0,1) using acceptance-rejection
    with a Laplace(0,1) proposal.
    """
    samples = []
    # c is chosen so that normal_pdf(x) <= c * laplace_pdf(x) for all x
    c = 1.32  # an empirically safe constant

    while len(samples) < n:
        # Sample from Laplace(0,1)
        x = np.random.laplace(loc=0.0, scale=1.0)
        u = np.random.rand()

        # PDF of N(0,1)
        normal_pdf = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-x**2 / 2.0)
        # PDF of Laplace(0,1)
        laplace_pdf = 0.5 * np.exp(-abs(x))

        # Accept if u < ratio (scaled by c)
        if u < normal_pdf / (c * laplace_pdf):
            samples.append(x)

    return np.array(samples)

def generate_normals_builtin(n):
    """
    Generate n samples from N(0,1) using NumPy's built-in generator.
    """
    return np.random.randn(n)


def load_mmm_returns(
    filename="stock_data.csv",
    date_column="Date",
    price_column="MMM",
    dayfirst=True
):

    try:

        df = pd.read_csv(filename, parse_dates=[date_column], dayfirst=dayfirst)

        df.sort_values(by=date_column, inplace=True)

        df[price_column] = df[price_column].astype(float)

        df["MMM_Return"] = np.log(df[price_column] / df[price_column].shift(1))

        returns = df["MMM_Return"].dropna().values

        return returns

    except Exception as e:
        print(f"Error loading {filename}: {e}")
        print("Falling back on synthetic normal data.")
        return np.random.randn(1000)

# -------------------------------
# Problem 4
# -------------------------------
def main():
    np.random.seed(42) 
    
    n_samples = 1000

    # 1. Box-Müller
    samples_box_muller = generate_normals_box_muller(n_samples)

    # 2. Acceptance-Rejection
    samples_accept_reject = generate_normals_accept_reject(n_samples)

    # 3. Built-in Generator
    samples_builtin = generate_normals_builtin(n_samples)

    # 4. Load MMM Data & Compute Returns
    samples_mmm = load_mmm_returns(
        filename="stock_data.csv",
        date_column="Date",
        price_column="MMM",
        dayfirst=True
    )

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Q-Q plot: Box-Müller
    stats.probplot(samples_box_muller, dist="norm", plot=axs[0, 0])
    axs[0, 0].set_title("Q-Q Plot: Box-Müller Transform")

    # Q-Q plot: Acceptance-Rejection
    stats.probplot(samples_accept_reject, dist="norm", plot=axs[0, 1])
    axs[0, 1].set_title("Q-Q Plot: Acceptance-Rejection")

    # Q-Q plot: Built-in
    stats.probplot(samples_builtin, dist="norm", plot=axs[1, 0])
    axs[1, 0].set_title("Q-Q Plot: Built-in Generator")

    # Q-Q plot: MMM Stock Returns
    stats.probplot(samples_mmm, dist="norm", plot=axs[1, 1])
    axs[1, 1].set_title("Q-Q Plot: MMM Stock Returns")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
