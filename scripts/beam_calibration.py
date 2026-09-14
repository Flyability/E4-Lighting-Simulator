import numpy as np
from scipy.optimize import minimize_scalar

# Measured data: distance from center (cm) and measured lux
distances = np.array([0, 10, 20, 30, 40])  # cm
lux_measured = np.array([4749, 4366, 3540, 2436, 1510])

# Normalize relative to center
lux_norm = lux_measured / lux_measured[0]

print("=== MEASURED DATA ANALYSIS ===")
print(f"Distance (cm) | Measured lux | Normalized")
for d, lux, norm in zip(distances, lux_measured, lux_norm):
    print(f"{d:13} | {lux:12.0f} | {norm:12.3f}")

# Assume wall distance (default in simulator)
wall_distance = 50  # cm

# Calculate angles and distances for each point
angles_deg = np.degrees(np.arctan(distances / wall_distance))
radial_distances = np.sqrt(wall_distance**2 + distances**2)
cos_theta = np.cos(np.radians(angles_deg))
distance_factor = (wall_distance / radial_distances) ** 2

print(f"\n=== GEOMETRY (wall at {wall_distance} cm) ===")
print(f"Offset (cm) | LED distance (cm) | Angle (°) | cos(θ)")
for d, r, angle, cos_t in zip(distances, radial_distances, angles_deg, cos_theta):
    print(f"{d:11} | {r:17.1f} | {angle:10.1f} | {cos_t:6.3f}")

# Model: I(θ) ∝ cos^(n+1)(θ) × (d/r)²
# The exponent is (n+1) because cos^n comes from the LED emission pattern
# and cos^1 comes from the incidence angle on the wall surface

def beam_model(n_plus_1, angles_rad, dist_factor):
    """Calculate relative intensity with cos^(n+1)(θ) × (d/r)² model"""
    return (np.cos(angles_rad) ** n_plus_1) * dist_factor

def error_function(n_plus_1):
    """Calculate mean squared error between model and measurements"""
    angles_rad = np.radians(angles_deg)
    model_values = beam_model(n_plus_1, angles_rad, distance_factor)
    sse = np.sum((lux_norm - model_values) ** 2)
    return sse

# Find optimal n+1
result = minimize_scalar(error_function, bounds=(1.5, 5.0), method='bounded')
optimal_n_plus_1 = result.x
optimal_n = optimal_n_plus_1 - 1.0

print(f"\n=== BEAM PROFILE OPTIMIZATION ===")
print(f"Optimal exponent (n+1): {optimal_n_plus_1:.2f}")
print(f"LED pattern exponent (n): {optimal_n:.2f}")

# Calculate optimal model values
angles_rad = np.radians(angles_deg)
model_optimal = beam_model(optimal_n_plus_1, angles_rad, distance_factor)

print(f"\n=== MODEL vs MEASURED COMPARISON ===")
print(f"Offset (cm) | Measured | Model (n={optimal_n:.2f}) | Error %")
for d, meas, mod in zip(distances, lux_norm, model_optimal):
    error_pct = abs(meas - mod) / meas * 100
    print(f"{d:11} | {meas:8.3f} | {mod:23.3f} | {error_pct:8.1f}%")

# In the simulator, n is calculated from:
# n_base = ln(0.5) / ln(cos(viewing_angle/2))
# n = n_base × (1 + uniformity × 2)

# For viewing_angle = 120°:
viewing_angle = 120  # degrees
theta_half = np.radians(viewing_angle / 2)
n_base = np.log(0.5) / np.log(np.cos(theta_half))

print(f"\n=== SIMULATOR PARAMETERS ===")
print(f"Current viewing angle: {viewing_angle}°")
print(f"n_base (from viewing angle): {n_base:.3f}")

# To obtain optimal n, calculate required uniformity
if abs(optimal_n - n_base) < 0.01:
    uniformity_needed = 0.0
    print(f"\n✓ Optimal n ({optimal_n:.2f}) already matches n_base!")
    print(f"  → Set Focus factor = 0.0")
else:
    # n = n_base × (1 + uniformity × 2)
    # optimal_n = n_base × (1 + uniformity × 2)
    # uniformity = (optimal_n/n_base - 1) / 2
    uniformity_needed = (optimal_n / n_base - 1.0) / 2.0
    print(f"\n→ To obtain n = {optimal_n:.2f}, set:")
    print(f"  Focus factor = {uniformity_needed:.3f}")
    
if uniformity_needed < 0:
    print(f"\n⚠ WARNING: The calculated value is negative!")
    print(f"  The simulator accepts Focus factor between 0.0 and 1.0")
    print(f"  You may need to change the viewing angle")
    
    # Calculate alternative viewing angle that would give optimal n with uniformity=0
    # n = ln(0.5) / ln(cos(θ_half))
    # cos(θ_half) = exp(ln(0.5) / n)
    theta_half_needed = np.arccos(np.exp(np.log(0.5) / optimal_n))
    viewing_angle_needed = 2 * np.degrees(theta_half_needed)
    print(f"  ALTERNATIVE: Set Viewing angle = {viewing_angle_needed:.0f}° and Focus factor = 0.0")

# Test with common n values
print(f"\n=== COMPARISON WITH STANDARD PROFILES ===")
test_n_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
print(f"n    | RMS Error | Focus factor")
print(f"-----|------------|-------------")
for n_test in test_n_values:
    model_test = beam_model(n_test + 1, angles_rad, distance_factor)
    rmse = np.sqrt(np.mean((lux_norm - model_test) ** 2))
    uniformity_test = (n_test / n_base - 1.0) / 2.0
    print(f"{n_test:.1f}  | {rmse:10.4f} | {uniformity_test:12.3f}")

print(f"\n✓ Optimal: n = {optimal_n:.2f} (RMS Error = {np.sqrt(result.fun / len(distances)):.4f})")
