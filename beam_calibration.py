import numpy as np
from scipy.optimize import minimize_scalar

# Dati misurati: distanza dal centro (cm) e lux misurati
distances = np.array([0, 10, 20, 30, 40])  # cm
lux_measured = np.array([4749, 4366, 3540, 2436, 1510])

# Normalizza rispetto al centro
lux_norm = lux_measured / lux_measured[0]

print("=== ANALISI DATI MISURATI ===")
print(f"Distanza (cm) | Lux misurati | Normalizzato")
for d, lux, norm in zip(distances, lux_measured, lux_norm):
    print(f"{d:13} | {lux:12.0f} | {norm:12.3f}")

# Assumi distanza muro (default nel simulator)
wall_distance = 50  # cm

# Calcola angoli e distanze per ogni punto
angles_deg = np.degrees(np.arctan(distances / wall_distance))
radial_distances = np.sqrt(wall_distance**2 + distances**2)
cos_theta = np.cos(np.radians(angles_deg))
distance_factor = (wall_distance / radial_distances) ** 2

print(f"\n=== GEOMETRIA (muro a {wall_distance} cm) ===")
print(f"Offset (cm) | Distanza LED (cm) | Angolo (°) | cos(θ)")
for d, r, angle, cos_t in zip(distances, radial_distances, angles_deg, cos_theta):
    print(f"{d:11} | {r:17.1f} | {angle:10.1f} | {cos_t:6.3f}")

# Modello: I(θ) ∝ cos^(n+1)(θ) × (d/r)²
# L'esponente è (n+1) perché cos^n viene dal pattern di emissione LED
# e cos^1 viene dall'angolo di incidenza sulla superficie del muro

def beam_model(n_plus_1, angles_rad, dist_factor):
    """Calcola intensità relativa con modello cos^(n+1)(θ) × (d/r)²"""
    return (np.cos(angles_rad) ** n_plus_1) * dist_factor

def error_function(n_plus_1):
    """Calcola errore quadratico medio tra modello e misure"""
    angles_rad = np.radians(angles_deg)
    model_values = beam_model(n_plus_1, angles_rad, distance_factor)
    sse = np.sum((lux_norm - model_values) ** 2)
    return sse

# Trova n+1 ottimale
result = minimize_scalar(error_function, bounds=(1.5, 5.0), method='bounded')
optimal_n_plus_1 = result.x
optimal_n = optimal_n_plus_1 - 1.0

print(f"\n=== OTTIMIZZAZIONE PROFILO BEAM ===")
print(f"Esponente ottimale (n+1): {optimal_n_plus_1:.2f}")
print(f"Esponente LED pattern (n): {optimal_n:.2f}")

# Calcola valori del modello ottimale
angles_rad = np.radians(angles_deg)
model_optimal = beam_model(optimal_n_plus_1, angles_rad, distance_factor)

print(f"\n=== CONFRONTO MODELLO vs MISURATO ===")
print(f"Offset (cm) | Misurato | Modello (n={optimal_n:.2f}) | Errore %")
for d, meas, mod in zip(distances, lux_norm, model_optimal):
    error_pct = abs(meas - mod) / meas * 100
    print(f"{d:11} | {meas:8.3f} | {mod:23.3f} | {error_pct:8.1f}%")

# Nel simulatore, n è calcolato da:
# n_base = ln(0.5) / ln(cos(viewing_angle/2))
# n = n_base × (1 + uniformity × 2)

# Per viewing_angle = 120°:
viewing_angle = 120  # gradi
theta_half = np.radians(viewing_angle / 2)
n_base = np.log(0.5) / np.log(np.cos(theta_half))

print(f"\n=== PARAMETRI SIMULATORE ===")
print(f"Viewing angle corrente: {viewing_angle}°")
print(f"n_base (da viewing angle): {n_base:.3f}")

# Per ottenere n ottimale, calcola uniformity necessario
if abs(optimal_n - n_base) < 0.01:
    uniformity_needed = 0.0
    print(f"\n✓ L'n ottimale ({optimal_n:.2f}) corrisponde già a n_base!")
    print(f"  → Imposta Focus factor = 0.0")
else:
    # n = n_base × (1 + uniformity × 2)
    # optimal_n = n_base × (1 + uniformity × 2)
    # uniformity = (optimal_n/n_base - 1) / 2
    uniformity_needed = (optimal_n / n_base - 1.0) / 2.0
    print(f"\n→ Per ottenere n = {optimal_n:.2f}, imposta:")
    print(f"  Focus factor = {uniformity_needed:.3f}")
    
if uniformity_needed < 0:
    print(f"\n⚠ ATTENZIONE: Il valore calcolato è negativo!")
    print(f"  Il simulatore accetta Focus factor tra 0.0 e 1.0")
    print(f"  Potrebbe essere necessario modificare il viewing angle")
    
    # Calcola viewing angle alternativo che darebbe n ottimale con uniformity=0
    # n = ln(0.5) / ln(cos(θ_half))
    # cos(θ_half) = exp(ln(0.5) / n)
    theta_half_needed = np.arccos(np.exp(np.log(0.5) / optimal_n))
    viewing_angle_needed = 2 * np.degrees(theta_half_needed)
    print(f"  ALTERNATIVA: Imposta Viewing angle = {viewing_angle_needed:.0f}° e Focus factor = 0.0")

# Test con diversi valori di n comuni
print(f"\n=== CONFRONTO CON PROFILI STANDARD ===")
test_n_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
print(f"n    | Errore RMS | Focus factor")
print(f"-----|------------|-------------")
for n_test in test_n_values:
    model_test = beam_model(n_test + 1, angles_rad, distance_factor)
    rmse = np.sqrt(np.mean((lux_norm - model_test) ** 2))
    uniformity_test = (n_test / n_base - 1.0) / 2.0
    print(f"{n_test:.1f}  | {rmse:10.4f} | {uniformity_test:12.3f}")

print(f"\n✓ Ottimale: n = {optimal_n:.2f} (Errore RMS = {np.sqrt(result.fun / len(distances)):.4f})")
