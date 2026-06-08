import numpy as np

# ----- PARAMETERS -----
N = 20000  # Lookup table size for arcsin approximation
PI2 = np.pi / 2

# ----- SINE LOOKUP TABLE -----
x_table = np.linspace(0, PI2, N+1)
sine_table = np.sin(x_table)

def fast_sin(x):
    """Fast sine using table lookup (0..π/2 only)."""
    x = np.clip(x, 0, PI2)
    idx = (x / PI2 * N).astype(np.int32)
    idx = np.clip(idx, 0, N)
    return sine_table[idx]

def fast_arcsin(y):
    """Fast arcsin using lookup table search (0..1 only)."""
    y = np.clip(y, 0, 1)
    if np.isscalar(y):
        idx = np.abs(sine_table - y).argmin()
        return idx * PI2 / N
    else:
        idxs = np.abs(sine_table[:, None] - y[None, :]).argmin(axis=0)
        return idxs * PI2 / N

# ----- CORE: COS FROM SINE (π/2 SHIFT) -----
def cos_from_sin(x):
    return fast_sin(PI2 - x)

# ----- ADDITION (Tier 0 √2, π/4 identity) -----
def sine_add(A, B):
    K = A + B
    K = np.where(K == 0, 1, K)
    a = A / K
    x = fast_arcsin(np.sqrt(a))
    sin_x = fast_sin(x)
    sin_x_p4 = fast_sin(x + (np.pi / 4))
    b = (np.sqrt(2) * sin_x_p4 - sin_x) ** 2
    return a * K, b * K, K

# ----- SUBTRACTION -----
def sine_subtract(A, B):
    K = np.abs(A) + np.abs(B)
    K = np.where(K == 0, 1, K)
    a = np.abs(A) / K
    b = np.abs(B) / K
    diff = (a - b) * K
    sign = np.sign(A - B)
    return sign * np.abs(diff)

# ----- MULTIPLICATION -----
def sine_multiply(A, B, max_value):
    a = A / max_value
    b = B / max_value
    x = fast_arcsin(np.sqrt(a))
    y = fast_arcsin(np.sqrt(b))
    sin_x = fast_sin(x)
    sin_y = fast_sin(y)
    prod = sin_x * sin_y
    return (prod ** 2) * max_value

# ----- DIVISION -----
def sine_divide(A, B, max_value):
    a = A / max_value
    b = B / max_value
    b = np.where(b == 0, 1e-9, b)
    x = fast_arcsin(np.sqrt(a))
    y = fast_arcsin(np.sqrt(b))
    sin_x = fast_sin(x)
    sin_y = fast_sin(y)
    numerator = sin_x * sin_y
    denominator = sin_y ** 2
    quotient = numerator / denominator
    return (quotient ** 2) * max_value

# ----- SQUARE ROOT -----
def sine_sqrt(A, max_value):
    a = A / max_value
    x = fast_arcsin(np.sqrt(a))
    sin_x = fast_sin(x)
    root = np.abs(sin_x)
    return (root ** 2) * max_value

# ----- DEMO -----
if __name__ == "__main__":
    A = np.array([37.0, 0.5, 80.0, 13.0])
    B = np.array([63.0, 0.5, 20.0, 29.0])
    max_value = np.maximum(A+B, 1)

    # ADDITION
    Ares, Bres, Kres = sine_add(A, B)
    print("\nSine-only Addition:")
    for i in range(len(A)):
        print(f"{A[i]} + {B[i]} = {Kres[i]} (A': {Ares[i]}, B': {Bres[i]})")

    # SUBTRACTION
    Subres = sine_subtract(A, B)
    print("\nSine-only Subtraction:")
    for i in range(len(A)):
        print(f"{A[i]} - {B[i]} ≈ {Subres[i]}")

    # MULTIPLICATION
    Mres = sine_multiply(A, B, max_value)
    print("\nSine-only Multiplication:")
    for i in range(len(A)):
        print(f"{A[i]} * {B[i]} ≈ {Mres[i]}")

    # DIVISION
    Dres = sine_divide(A, B, max_value)
    print("\nSine-only Division:")
    for i in range(len(A)):
        print(f"{A[i]} / {B[i]} ≈ {Dres[i]}")

    # SQUARE ROOT
    Sqrtres = sine_sqrt(A, max_value)
    print("\nSine-only Square Root:")
    for i in range(len(A)):
        print(f"sqrt({A[i]}) ≈ {Sqrtres[i]}")