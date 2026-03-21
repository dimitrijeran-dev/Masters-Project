
import numpy as np
import pandas as pd

def load_ki_csv(path):
    df = pd.read_csv(path)
    if "a" not in df.columns or "KI" not in df.columns:
        raise ValueError("CSV must contain columns: a, KI")
    return df.sort_values("a")

def compute_delta_k_from_R(Kmax, R):
    return Kmax * (1 - R)

def compute_delta_k(Kmax, Kmin):
    return Kmax - Kmin

def paris_law(delta_k, C, m):
    return C * (delta_k ** m)

def integrate_crack_growth(a, delta_k, C, m):
    dadn = paris_law(delta_k, C, m)
    da = np.gradient(a)
    dN = da / dadn
    N = np.cumsum(dN)
    return N, dadn
