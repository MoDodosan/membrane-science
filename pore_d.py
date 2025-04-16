import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd

# Input data: solute radius r_s and observed rejection R
r_s = np.array([0.2, 0.26, 0.32, 0.401, 0.471, 0.59])  # Solute radius [nm]
R = np.array([0.0417, 0.1523, 0.6252, 0.8762, 0.9545, 0.9848])  # Rejection (0 to 1)

# Convert rejection to corresponding standard normal quantile y
y = norm.ppf(R)  # Inverse CDF (probit function)

# Step 1: Linear fit to y = A + B * ln(r_s)
def linear_model_1(ln_r_s, A, B):
    return A + B * ln_r_s

# Compute ln(r_s)
ln_r_s = np.log(r_s)

# Fit A and B
params_1, _ = curve_fit(linear_model_1, ln_r_s, y)
A, B = params_1
print("Fitted parameter A =", A)
print("Fitted parameter B =", B)

# Step 2: Fit using equation:
# y = (ln(r_s) - ln(average_pore_size)) / ln(geometric_std_dev)
def linear_model_2(y, ln_avg_pore, ln_geo_std):
    return ln_avg_pore + y * ln_geo_std

# Fit ln(r_s) vs y
params_2, _ = curve_fit(linear_model_2, y, ln_r_s)
ln_avg_pore, ln_geo_std = params_2
print("Fitted ln(average pore size) =", ln_avg_pore)
print("Fitted ln(geometric std. dev.) =", ln_geo_std)

# Convert log values to actual parameters
average_pore_size = np.exp(ln_avg_pore)
geometric_std_dev = np.exp(ln_geo_std)
print("Average pore size =", average_pore_size)
print("Geometric standard deviation =", geometric_std_dev)

# Generate pore size range for plotting (in nm)
pore_radii = np.linspace(0.1, 1.0, 100)

# Define log-normal PDF
def log_normal_pdf(r_p, mean, sigma):
    return (1 / (r_p * np.log(sigma) * np.sqrt(2 * np.pi))) * np.exp(
        -((np.log(r_p) - np.log(mean)) ** 2) / (2 * (np.log(sigma) ** 2))
    )

# Calculate distribution
pore_distribution = log_normal_pdf(pore_radii, average_pore_size, geometric_std_dev)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(pore_radii, pore_distribution, label="Pore Size Distribution", color="blue")
plt.xlabel("Pore Radius (nm)")
plt.ylabel("Probability Density")
plt.title("Log-Normal Pore Size Distribution")
plt.legend()
plt.grid(True)
plt.show()

# Save results to Excel
data = pd.DataFrame({
    "Pore Radius (nm)": pore_radii,
    "Probability Density": pore_distribution
})
data.to_excel("pore_size_distribution_NF270EDA.xlsx", index=False)
print("Pore size distribution data saved to 'pore_size_distribution_NF270EDA.xlsx'.")