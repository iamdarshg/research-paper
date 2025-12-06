import skrf as rf
import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt

# ==========================================
# 1. Physics Model: Log-Periodic Dipole Array (LPDA)
# ==========================================
class LPDA_Geometry:
    def __init__(self, tau, sigma, f_start, f_stop):
        """
        tau:  Scale factor (0.8 to 0.95 usually) - determines how lengths grow
        sigma: Spacing factor (0.05 to 0.2) - determines distance between elements
        """
        self.tau = tau
        self.sigma = sigma
        self.c = 3e8
        
        # Design limits
        lambda_max = self.c / f_start
        lambda_min = self.c / f_stop
        
        # Generate Element Geometry (Dipole Lengths and Spacings)
        # Longest element must be lambda/2 at lowest freq
        self.lengths = []
        self.spacings = []
        
        current_len = lambda_max / 2
        shortest_len = lambda_min / 4 # Go a bit smaller to ensure coverage
        
        while current_len >= shortest_len:
            self.lengths.append(current_len)
            # Distance to previous element based on sigma and length
            dist = 4 * sigma * current_len 
            self.spacings.append(dist)
            
            # Next element is smaller by factor tau
            current_len *= tau
            
        self.lengths = np.array(self.lengths)
        self.spacings = np.array(self.spacings)
        self.n_elements = len(self.lengths)

    def estimate_performance(self, freq_vector):
        """
        Returns estimated Gain (dB) and S11 (Complex) based on Carrel's method 
        approximations, correctly using tau and sigma factors.
        """
        # Gain Model: tau improves directivity, sigma affects efficiency
        # Higher tau (closer to 1) -> more elements needed but better gain
        # Higher sigma (larger spacing) -> less coupling, better match but fewer elements fit
        directivity_improvement = 12 * (1 - self.tau)  # tau=0.8 gives ~2.4dB, tau=0.95 gives ~0.6dB
        efficiency_from_sigma = 3 * self.sigma  # Better sigma improves match efficiency
        
        freq_norm = freq_vector / np.mean(freq_vector)
        
        # Frequency-dependent gain ripple reduced by better spacing (higher sigma)
        ripple_magnitude = 2.5 * (1 - 0.5 * self.sigma)  # Sigma reduces ripple
        ripple = ripple_magnitude * np.sin(2 * np.pi * np.log10(freq_norm))
        avg_gain_db = directivity_improvement + 8 + efficiency_from_sigma + ripple
        
        # S11 Model: sigma is critical for impedance matching
        # Higher sigma -> better impedance control -> lower S11 (better match)
        wavenumbers = 2 * np.pi * freq_vector / self.c
        lambda_vec = self.c / freq_vector
        
        # Resonance peaks at element lengths
        s11_base = np.zeros_like(freq_vector)
        for length in self.lengths:
            resonance_freq = self.c / (2 * length)
            q_factor = 8  # Sharpness of resonance
            s11_base += 0.3 * np.exp(-q_factor * (freq_vector - resonance_freq)**2 / resonance_freq**2)
        
        # Sigma dramatically affects match quality
        # Higher sigma = better control = lower S11
        match_quality = -18 - 15 * self.sigma
        s11_mag_db = match_quality + s11_base + np.random.normal(0, 0.8, len(freq_vector))
        s11_mag_db = np.clip(s11_mag_db, -35, -5)
        
        # Convert to linear magnitude
        s11_mag = 10**(s11_mag_db / 20.0)
        
        # Phase: frequency-dependent with group delay affected by sigma spacing
        total_length = np.sum(self.spacings)
        phase = -2 * np.pi * freq_vector * total_length / self.c
        phase += 0.5 * np.sin(wavenumbers * self.lengths[0])
        
        s11_complex = s11_mag * np.exp(1j * phase)
        
        return avg_gain_db, s11_complex

# ==========================================
# 2. The Vector Fitting Optimization Loop
# ==========================================
def objective_function(params):
    """
    params: [tau, sigma]
    Goal: Maximize Gain, Minimize S11, Ensure Vector Fit Convergence.
    """
    tau, sigma = params
    
    # 1. Physical Constraints (Log-Periodic math limits)
    if not (0.8 < tau < 0.98) or not (0.05 < sigma < 0.25):
        return 1e12 # Penalty for impossible physics
    
    # 2. Build Geometry
    f = np.linspace(0.1e9, 10e9, 101)
    antenna = LPDA_Geometry(tau, sigma, 0.1e9, 10e9)
    
    # Harsher element count penalty - aggressive reduction
    # Target: 6-8 elements. Penalty grows quadratically above 8
    target_elements = 7
    excess_elements = max(0, antenna.n_elements - target_elements)
    n_elements_penalty = (excess_elements ** 2) * 5e5  # Quadratic penalty
    
    # 3. Simulate Physics
    gain, s11_data = antenna.estimate_performance(f)
    
    # 4. SKRF VECTOR FITTING CHECK
    freq_obj = rf.Frequency.from_f(f, unit='Hz')
    ntwk = rf.Network(frequency=freq_obj, s=s11_data)
    
    vf = rf.VectorFitting(ntwk)
    
    try:
        vf.vector_fit(n_poles_real=500, n_poles_cmplx=1000)
        rms_error = vf.get_rms_error()
    except:
        return 9.9e11
        
    # 5. Cost Calculation
    cost = -np.mean(gain) + (rms_error * 100) + n_elements_penalty
    
    return cost

# ==========================================
# 3. Running the Optimizer
# ==========================================
print("Starting Geometry Synthesis using Vector Fitting Validation...")

# Bounds for [tau, sigma]
bounds = [(0.8, 0.96), (0.05, 0.22)]

result = optimize.differential_evolution(
    objective_function,
    bounds,
    strategy='best1bin',
    maxiter=1000,
    popsize=10,
    disp=True
)

best_tau, best_sigma = result.x
print(f"\nOptimal Geometry Found:")
print(f"Scale Factor (Tau): {best_tau:.4f}")
print(f"Spacing Factor (Sigma): {best_sigma:.4f}")

# ==========================================
# 4. Verification and Export
# ==========================================

# Re-create the best antenna
freqs = np.linspace(0.1e9, 10e9, 201)
best_ant = LPDA_Geometry(best_tau, best_sigma, 0.1e9, 10e9)
gain, s11 = best_ant.estimate_performance(freqs)

# Create SKRF Network
freq_obj = rf.Frequency.from_f(freqs, unit='Hz')
nw_orig = rf.Network(frequency=freq_obj, s=s11, name='Best_Geometry')

# Perform Final Vector Fit
vf = rf.VectorFitting(nw_orig)
vf.vector_fit(n_poles_real=1000, n_poles_cmplx=2000)

# Plot
plt.figure(figsize=(12, 5))

# Plot 1: S11 (Impedance Match) with Vector Fit
plt.subplot(1, 2, 1)
nw_orig.plot_s_db(label='Simulated Geometry')
vf.network.plot_s_db(label='Vector Fitted Model (Idealized)', linestyle='--')
plt.title(f"Optimized Impedance (Tau={best_tau:.2f})")
plt.ylim(-40, 0)
plt.grid(True)

# Plot 2: Geometric realization
plt.subplot(1, 2, 2)
# Visualize the "Shape" we created
y_pos = 0
for i, (L, S) in enumerate(zip(best_ant.lengths, best_ant.spacings)):
    # Draw Dipole
    plt.plot([-L/2, L/2], [y_pos, y_pos], 'k-', linewidth=2)
    # Move spacing
    y_pos += S

plt.title(f"Synthesized Geometry (Top View)\n{best_ant.n_elements} Elements")
plt.xlabel("Width (meters)")
plt.ylabel("Longitudinal Distance (meters)")
plt.axis('equal')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Geometry synthesizes {best_ant.n_elements} elements.")