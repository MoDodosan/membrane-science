from pyomo.environ import (
    ConcreteModel, Var, Param, Set, RangeSet, Constraint, Expression,
    Objective, SolverFactory, Reals, NonNegativeReals, value, exp, minimize
)
from math import log, pi

# =========================================================
# 0. Mechanism switches
#    1 = enable, 0 = disable
# =========================================================
USE_STERIC = 1
USE_DONNAN = 1
USE_DE = 1

# =========================================================
# 1. Create model
# =========================================================
model = ConcreteModel()

# =========================================================
# 2. Basic constants and membrane parameters
# =========================================================
L = 100e-9
N_NODE = 11
dx = L / (N_NODE - 1)

model.F = Param(initialize=96485.0)            # C/mol
model.R = Param(initialize=8.314)              # J/(mol K)
model.T = Param(initialize=298.15)             # K
model.r_p = Param(initialize=3.62e-9)          # m
model.eta = Param(initialize=1.0e-3)           # Pa s
model.tau = Param(initialize=0.9)
model.epsilon = Param(initialize=0.12)
model.sigma = Param(initialize=-2.2e-3)        # C/m^2
model.area = Param(initialize=0.000804096)     # m^2
model.delta_P = Param(initialize=5.0e6)        # Pa

# dielectric properties
model.eps0 = Param(initialize=8.854e-12)       # F/m
model.e_charge = Param(initialize=1.602e-19)   # C
model.NA = Param(initialize=6.02214076e23)     # 1/mol

# bulk solvent dielectric constant
model.eps_w = Param(initialize=78.5)

# interfacial single water layer
model.eps_star = Param(initialize=31.0)
model.delta = Param(initialize=0.28e-9)        # m

# boundary layer thicknesses
model.delta_f = Param(initialize=50e-6)        # m
model.delta_p = Param(initialize=50e-6)        # m

def L_eff_rule(m):
    return m.tau * L / m.epsilon

model.L_eff = Expression(rule=L_eff_rule)

def eps_p_rule(m):
    return m.eps_star + (m.eps_w - m.eps_star) * (1.0 - m.delta / m.r_p) ** 2

model.eps_p = Expression(rule=eps_p_rule)

# =========================================================
# 3. Ion system
# =========================================================
ions = ['H+', 'Li+', 'Ni++', 'Co++', 'SO4--']

z_ions = {
    'H+': 1,
    'Li+': 1,
    'Ni++': 2,
    'Co++': 2,
    'SO4--': -2
}

# feed bulk concentration [mol/m^3]
c_feed_bulk = {
    'H+': 0.01e3,
    'Li+': 0.112e3,
    'Ni++': 0.100e3,
    'Co++': 0.070e3,
    'SO4--': 0.231e3
}

# experimental permeate bulk concentration [mol/m^3]
c_perm_bulk_exp = {
    'H+': 0.01e3,
    'Li+': 0.04e3,
    'Ni++': 0.00046e3,
    'Co++': 0.0003e3,
    'SO4--': 0.02576e3
}

# diffusion coefficients [m^2/s]
D_ions = {
    'H+': 9.31e-9,
    'Li+': 1.67e-9,
    'Ni++': 0.661e-9,
    'Co++': 0.732e-9,
    'SO4--': 1.06e-9
}

# ion radii [m]
r_ion = {
    'H+': 2.82e-10,
    'Li+': 3.82e-10,
    'Ni++': 4.04e-10,
    'Co++': 4.23e-10,
    'SO4--': 3.97e-10
}

model.ions = Set(initialize=ions)
model.x = RangeSet(0, N_NODE - 1)
model.seg = RangeSet(0, N_NODE - 2)
model.x_internal = RangeSet(1, N_NODE - 2)

model.z = Param(model.ions, initialize=z_ions)
model.c_feed_bulk = Param(model.ions, initialize=c_feed_bulk)
model.c_perm_bulk_exp = Param(model.ions, initialize=c_perm_bulk_exp)
model.D = Param(model.ions, initialize=D_ions)
model.r_i = Param(model.ions, initialize=r_ion)

# =========================================================
# 4. Hindrance factors and steric partition
# =========================================================
def calc_Kd(lambda_val):
    if lambda_val <= 0.95:
        return (
            1
            + 9.0 / 8.0 * lambda_val * log(lambda_val)
            - 1.56034 * lambda_val
            + 0.528155 * lambda_val**2
            + 1.91521 * lambda_val**3
            - 2.81903 * lambda_val**4
            + 0.270788 * lambda_val**5
            + 1.10115 * lambda_val**6
            - 0.435933 * lambda_val**7
        ) / ((1.0 - lambda_val) ** 2)
    else:
        return 0.984 * ((1.0 - lambda_val) / lambda_val) ** 2.5

def calc_Ka(lambda_val):
    numerator = 1.0 + 3.867 * lambda_val - 1.907 * lambda_val**2 - 0.834 * lambda_val**3
    denominator = 1.0 + 1.867 * lambda_val
    return numerator / denominator

def calc_phi_steric(lambda_val):
    if lambda_val >= 1.0:
        return 1e-12
    return (1.0 - lambda_val) ** 2

rp_val = value(model.r_p)

Kd_dict = {}
Ka_dict = {}
phi_steric_dict = {}

for ion in ions:
    lam = r_ion[ion] / rp_val
    if USE_STERIC == 1:
        Kd_dict[ion] = calc_Kd(lam)
        Ka_dict[ion] = calc_Ka(lam)
        phi_steric_dict[ion] = calc_phi_steric(lam)
    else:
        Kd_dict[ion] = 1.0
        Ka_dict[ion] = 1.0
        phi_steric_dict[ion] = 1.0

model.K_d = Param(model.ions, initialize=Kd_dict)
model.K_a = Param(model.ions, initialize=Ka_dict)
model.phi_steric = Param(model.ions, initialize=phi_steric_dict)

# =========================================================
# 5. Fixed charge density inside pore
# =========================================================
def Xd_rule(m):
    return 2.0 * m.sigma / (m.F * m.r_p)

model.X_d = Expression(rule=Xd_rule)

# =========================================================
# 6. Mass transfer coefficients
# =========================================================
def kf_rule(m, ion):
    return m.D[ion] / m.delta_f

def kp_rule(m, ion):
    return m.D[ion] / m.delta_p

model.k_f = Expression(model.ions, rule=kf_rule)
model.k_p = Expression(model.ions, rule=kp_rule)

# =========================================================
# 7. Dielectric exclusion (Born energy)
# =========================================================
def deltaG_de_rule(m, ion):
    if USE_DE == 1:
        return (
            m.NA * (m.z[ion] ** 2) * (m.e_charge ** 2)
            / (8.0 * pi * m.eps0 * m.r_i[ion])
            * (1.0 / m.eps_p - 1.0 / m.eps_w)
        )
    return 0.0

model.deltaG_DE = Expression(model.ions, rule=deltaG_de_rule)

def de_factor_rule(m, ion):
    return exp(-m.deltaG_DE[ion] / (m.R * m.T))

model.DE_factor = Expression(model.ions, rule=de_factor_rule)

# =========================================================
# 8. Variables
# =========================================================
# pore concentration profile
model.c = Var(model.ions, model.x, domain=NonNegativeReals, bounds=(1e-12, None))

# pore electric potential profile
model.psi = Var(model.x, domain=Reals)

# predicted ion flux
model.Ji = Var(model.ions, domain=Reals)

# water flux
model.Jv = Var(domain=NonNegativeReals, bounds=(1e-8, 1e-4), initialize=1.0e-5)

# feed-side membrane-interface concentrations
model.c_fm = Var(model.ions, domain=NonNegativeReals, bounds=(1e-12, None))

# permeate-side membrane-interface concentrations
model.c_pm = Var(model.ions, domain=NonNegativeReals, bounds=(1e-12, None))

# predicted permeate bulk concentration
model.c_perm_bulk_model = Var(model.ions, domain=NonNegativeReals, bounds=(1e-12, None))

# Donnan potentials
if USE_DONNAN == 1:
    model.psi_D_f = Var(domain=Reals, bounds=(-1.0, 1.0), initialize=-1.0e-2)
    model.psi_D_p = Var(domain=Reals, bounds=(-1.0, 1.0), initialize=-1.0e-2)
else:
    model.psi_D_f = Var(domain=Reals, bounds=(0.0, 0.0), initialize=0.0)
    model.psi_D_p = Var(domain=Reals, bounds=(0.0, 0.0), initialize=0.0)

# pore gradients
model.dc_dx = Var(model.ions, model.seg, domain=Reals, initialize=0.0)
model.dpsi_dx = Var(model.seg, domain=Reals, initialize=0.0)

# feed-side interface electric potential gradient
model.dpsi_dx_feed_interface = Var(domain=Reals, initialize=0.0)

# =========================================================
# 9. Initialization
# =========================================================
Jv_guess = 1.0e-5
psiD_feed_guess = -1.0e-2 if USE_DONNAN == 1 else 0.0
psiD_perm_guess = -1.0e-2 if USE_DONNAN == 1 else 0.0

for ion in ions:
    model.c_fm[ion].set_value(c_feed_bulk[ion])
    model.c_pm[ion].set_value(c_perm_bulk_exp[ion])
    model.c_perm_bulk_model[ion].set_value(max(c_perm_bulk_exp[ion], 1e-12))
    model.Ji[ion].set_value(c_perm_bulk_exp[ion] * Jv_guess)

    c0_guess = (
        phi_steric_dict[ion]
        * c_feed_bulk[ion]
        * exp(-z_ions[ion] * value(model.F) * psiD_feed_guess / (value(model.R) * value(model.T)))
        * value(model.DE_factor[ion])
    )

    cL_guess = (
        phi_steric_dict[ion]
        * c_perm_bulk_exp[ion]
        * exp(-z_ions[ion] * value(model.F) * psiD_perm_guess / (value(model.R) * value(model.T)))
        * value(model.DE_factor[ion])
    )

    c0_guess = max(c0_guess, 1e-12)
    cL_guess = max(cL_guess, 1e-12)

    for j in range(N_NODE):
        frac = j / (N_NODE - 1)
        cij = (1.0 - frac) * c0_guess + frac * cL_guess
        model.c[ion, j].set_value(max(cij, 1e-12))

for j in range(N_NODE):
    model.psi[j].set_value(0.0)

model.dpsi_dx_feed_interface.set_value(0.0)

# =========================================================
# 10. Osmotic pressure difference and water flux
#     Use membrane-interface concentrations
# =========================================================
def delta_pi_rule(m):
    return m.R * m.T * sum(m.c_fm[i] - m.c_pm[i] for i in m.ions)

model.delta_pi = Expression(rule=delta_pi_rule)

def water_flux_rule(m):
    return m.Jv == (m.r_p**2 * (m.delta_P - m.delta_pi)) / (8.0 * m.eta * m.L_eff)

model.water_flux = Constraint(rule=water_flux_rule)

# =========================================================
# 11. Feed-side mass transfer / concentration polarization
#     Ji = -k_f (c_fm - c_feed_bulk)
#          + Jv * c_fm
#          - z_i c_fm D_i F/(RT) * dpsi_dx_feed_interface
# =========================================================
def cp_feed_flux_rule(m, ion):
    return m.Ji[ion] == (
        -m.k_f[ion] * (m.c_fm[ion] - m.c_feed_bulk[ion])
        + m.Jv * m.c_fm[ion]
        - (
            m.z[ion] * m.c_fm[ion] * m.D[ion] * m.F
            / (m.R * m.T)
        ) * m.dpsi_dx_feed_interface
    )

model.cp_feed_flux = Constraint(model.ions, rule=cp_feed_flux_rule)

# permeate-side surface concentration
# assume well-mixed permeate side
def perm_surface_rule(m, ion):
    return m.c_pm[ion] == m.c_perm_bulk_model[ion]

model.perm_surface = Constraint(model.ions, rule=perm_surface_rule)

# =========================================================
# 12. Interface electroneutrality
# =========================================================
def electroneutrality_feed_interface_rule(m):
    return sum(m.z[i] * m.c_fm[i] for i in m.ions) == 0.0

model.electroneutrality_feed_interface = Constraint(rule=electroneutrality_feed_interface_rule)

def electroneutrality_permeate_rule(m):
    return sum(m.z[i] * m.c_pm[i] for i in m.ions) == 0.0

model.electroneutrality_permeate = Constraint(rule=electroneutrality_permeate_rule)

# =========================================================
# 13. Interface partition with steric + Donnan + DE
# =========================================================
def donnan_feed_rule(m, ion):
    return m.c[ion, 0] == (
        m.phi_steric[ion]
        * m.c_fm[ion]
        * exp(-m.z[ion] * m.F * m.psi_D_f / (m.R * m.T))
        * m.DE_factor[ion]
    )

model.donnan_feed = Constraint(model.ions, rule=donnan_feed_rule)

def donnan_perm_rule(m, ion):
    return m.c[ion, N_NODE - 1] == (
        m.phi_steric[ion]
        * m.c_pm[ion]
        * exp(-m.z[ion] * m.F * m.psi_D_p / (m.R * m.T))
        * m.DE_factor[ion]
    )

model.donnan_perm = Constraint(model.ions, rule=donnan_perm_rule)

# =========================================================
# 13.5 Donnan closure
# =========================================================
def donnan_feed_closure_rule(m):
    return sum(
        m.z[i] * m.phi_steric[i] * m.c_fm[i]
        * exp(-m.z[i] * m.F * m.psi_D_f / (m.R * m.T))
        * m.DE_factor[i]
        for i in m.ions
    ) + m.X_d == 0.0

model.donnan_feed_closure = Constraint(rule=donnan_feed_closure_rule)

def donnan_perm_closure_rule(m):
    return sum(
        m.z[i] * m.phi_steric[i] * m.c_pm[i]
        * exp(-m.z[i] * m.F * m.psi_D_p / (m.R * m.T))
        * m.DE_factor[i]
        for i in m.ions
    ) + m.X_d == 0.0

model.donnan_perm_closure = Constraint(rule=donnan_perm_closure_rule)

# =========================================================
# 14. Internal electroneutrality
# =========================================================
def electroneutrality_rule(m, j):
    return sum(m.z[i] * m.c[i, j] for i in m.ions) + m.X_d == 0.0

model.electroneutrality = Constraint(model.x_internal, rule=electroneutrality_rule)

# =========================================================
# 15. Potential reference
# =========================================================
def psi_ref_rule(m):
    return m.psi[0] == 0.0

model.psi_ref = Constraint(rule=psi_ref_rule)

# =========================================================
# 16. Finite-difference gradient definitions
# =========================================================
def dc_dx_def_rule(m, ion, s):
    return m.dc_dx[ion, s] == (m.c[ion, s + 1] - m.c[ion, s]) / dx

model.dc_dx_def = Constraint(model.ions, model.seg, rule=dc_dx_def_rule)

def dpsi_dx_def_rule(m, s):
    return m.dpsi_dx[s] == (m.psi[s + 1] - m.psi[s]) / dx

model.dpsi_dx_def = Constraint(model.seg, rule=dpsi_dx_def_rule)

# =========================================================
# 17. Average pore concentration
# =========================================================
def c_pore_avg_rule(m, ion, s):
    return 0.5 * (m.c[ion, s] + m.c[ion, s + 1])

model.c_pore_avg = Expression(model.ions, model.seg, rule=c_pore_avg_rule)

# =========================================================
# 18. Flux decomposition inside pore
#     J_i = diffusive + convective + electromigration
# =========================================================
def diffusive_term_rule(m, ion, s):
    return -m.K_d[ion] * m.D[ion] * m.dc_dx[ion, s]

model.diffusive_term = Expression(model.ions, model.seg, rule=diffusive_term_rule)

def convective_term_rule(m, ion, s):
    return m.K_a[ion] * m.c_pore_avg[ion, s] * m.Jv

model.convective_term = Expression(model.ions, model.seg, rule=convective_term_rule)

def electromigration_term_rule(m, ion, s):
    return -(
        m.z[ion]
        * m.c_pore_avg[ion, s]
        * m.K_d[ion]
        * m.D[ion]
        * m.F
        / (m.R * m.T)
    ) * m.dpsi_dx[s]

model.electromigration_term = Expression(model.ions, model.seg, rule=electromigration_term_rule)

# =========================================================
# 19. Electric potential gradient inside membrane
# =========================================================
def electric_potential_rule(m, s):
    numerator = sum(
        m.z[i] * (m.convective_term[i, s] - m.Ji[i]) / (m.K_d[i] * m.D[i])
        for i in m.ions
    )

    denominator = sum(
        (m.z[i] ** 2) * m.c_pore_avg[i, s]
        for i in m.ions
    )

    return m.dpsi_dx[s] == (m.R * m.T / m.F) * numerator / denominator

model.electric_potential = Constraint(model.seg, rule=electric_potential_rule)

# =========================================================
# 20. Extended Nernst-Planck equation
# =========================================================
def nernst_planck_rule(m, ion, s):
    return m.Ji[ion] == (
        m.diffusive_term[ion, s]
        + m.convective_term[ion, s]
        + m.electromigration_term[ion, s]
    )

model.nernst_planck = Constraint(model.ions, model.seg, rule=nernst_planck_rule)

# =========================================================
# 21. Flux-permeate concentration relation
#     Ji = Jv * c_perm_bulk_model
# =========================================================
def flux_perm_relation_rule(m, ion):
    return m.Ji[ion] == m.Jv * m.c_perm_bulk_model[ion]

model.flux_perm_relation = Constraint(model.ions, rule=flux_perm_relation_rule)

# =========================================================
# 22. Contribution analysis expressions
#     Feed-side equivalent entry flux decomposition
# =========================================================
def c_feed_steric_only_rule(m, ion):
    return m.phi_steric[ion] * m.c_fm[ion]

model.c_feed_steric_only = Expression(model.ions, rule=c_feed_steric_only_rule)

def c_feed_steric_donnan_rule(m, ion):
    return (
        m.phi_steric[ion]
        * m.c_fm[ion]
        * exp(-m.z[ion] * m.F * m.psi_D_f / (m.R * m.T))
    )

model.c_feed_steric_donnan = Expression(model.ions, rule=c_feed_steric_donnan_rule)

def c_feed_full_rule(m, ion):
    return (
        m.phi_steric[ion]
        * m.c_fm[ion]
        * exp(-m.z[ion] * m.F * m.psi_D_f / (m.R * m.T))
        * m.DE_factor[ion]
    )

model.c_feed_full = Expression(model.ions, rule=c_feed_full_rule)

def J_steric_eq_rule(m, ion):
    return m.Jv * m.c_feed_steric_only[ion]

model.J_steric_eq = Expression(model.ions, rule=J_steric_eq_rule)

def J_steric_donnan_eq_rule(m, ion):
    return m.Jv * m.c_feed_steric_donnan[ion]

model.J_steric_donnan_eq = Expression(model.ions, rule=J_steric_donnan_eq_rule)

def J_full_eq_rule(m, ion):
    return m.Jv * m.c_feed_full[ion]

model.J_full_eq = Expression(model.ions, rule=J_full_eq_rule)

def J_donnan_increment_rule(m, ion):
    return m.J_steric_donnan_eq[ion] - m.J_steric_eq[ion]

model.J_donnan_increment = Expression(model.ions, rule=J_donnan_increment_rule)

def J_DE_increment_rule(m, ion):
    return m.J_full_eq[ion] - m.J_steric_donnan_eq[ion]

model.J_DE_increment = Expression(model.ions, rule=J_DE_increment_rule)

# =========================================================
# 23. Objective function
# =========================================================
weights = {}
for ion in ions:
    weights[ion] = 1.0 / max(c_perm_bulk_exp[ion] ** 2, 1e-20)

model.w_fit = Param(model.ions, initialize=weights)

def objective_rule(m):
    fit_term = sum(
        m.w_fit[i] * (m.c_perm_bulk_model[i] - m.c_perm_bulk_exp[i]) ** 2
        for i in m.ions
    )

    reg_term = 1e-12 * (
        sum(m.psi[j] ** 2 for j in m.x)
        + m.psi_D_f ** 2 + m.psi_D_p ** 2
        + m.dpsi_dx_feed_interface ** 2
    )

    return fit_term + reg_term

model.obj = Objective(rule=objective_rule, sense=minimize)

# =========================================================
# 24. Solve
# =========================================================
solver = SolverFactory('ipopt')
solver.options['max_iter'] = 10000
solver.options['tol'] = 1e-10
solver.options['print_level'] = 5

results = solver.solve(model, tee=True)

print("Solver status:", results.solver.status)
print("Termination condition:", results.solver.termination_condition)

if str(results.solver.termination_condition).lower() != "optimal":
    print("\nWARNING: Solver did not converge to an optimal solution.")
    print("Printed variable values may still be non-physical or only locally feasible.")

# =========================================================
# 25. Output
# =========================================================
print("\n==================== RESULTS ====================")
print(f"eps_p                  = {value(model.eps_p):.6f}")
print(f"X_d                    = {value(model.X_d):.6e} mol/m^3")
print(f"delta_pi               = {value(model.delta_pi):.6e} Pa")
print(f"Jv                     = {value(model.Jv):.6e} m/s")
print(f"psi_D_f                = {value(model.psi_D_f):.12e} V")
print(f"psi_D_p                = {value(model.psi_D_p):.12e} V")
print(f"dpsi_dx_feed_interface = {value(model.dpsi_dx_feed_interface):.12e} V/m")

print("\n--- Experimental vs model-predicted permeate concentrations [mol/m^3] ---")
for ion in model.ions:
    print(
        f"{ion:6s}: exp = {value(model.c_perm_bulk_exp[ion]):.6e}, "
        f"model = {value(model.c_perm_bulk_model[ion]):.6e}"
    )

print("\n--- Dielectric exclusion energy and factor ---")
for ion in model.ions:
    print(
        f"{ion:6s}: DeltaG_DE = {value(model.deltaG_DE[ion]):.6e} J/mol, "
        f"DE_factor = {value(model.DE_factor[ion]):.6e}"
    )

print("\n--- Predicted membrane flux Ji [mol/(m^2 s)] ---")
for ion in model.ions:
    print(f"{ion:6s}: Ji = {value(model.Ji[ion]):.6e}")

print("\n--- Feed-side equivalent flux decomposition [mol/(m^2 s)] ---")
for ion in model.ions:
    print(f"\nIon: {ion}")
    print(f"  Steric-only equivalent flux              = {value(model.J_steric_eq[ion]):.6e}")
    print(f"  Steric + Donnan equivalent flux          = {value(model.J_steric_donnan_eq[ion]):.6e}")
    print(f"  Full (Steric + Donnan + DE) eq. flux     = {value(model.J_full_eq[ion]):.6e}")
    print(f"  Donnan incremental contribution          = {value(model.J_donnan_increment[ion]):.6e}")
    print(f"  Dielectric exclusion incremental contrib = {value(model.J_DE_increment[ion]):.6e}")

print("\n--- Feed-side membrane surface concentrations c_fm [mol/m^3] ---")
for ion in model.ions:
    print(f"{ion:6s}: c_fm = {value(model.c_fm[ion]):.6e}")

print("\n--- Permeate-side membrane surface concentrations c_pm [mol/m^3] ---")
for ion in model.ions:
    print(f"{ion:6s}: c_pm = {value(model.c_pm[ion]):.6e}")

print("\n--- Pore concentration profile inside membrane [mol/m^3] ---")
for ion in model.ions:
    print(f"\n{ion}")
    for j in model.x:
        xj = j * dx
        print(f"x = {xj:.3e} m, c = {value(model.c[ion, j]):.6e}")

print("\n--- Pore potential profile inside membrane [V] ---")
for j in model.x:
    xj = j * dx
    print(f"x = {xj:.3e} m, psi = {value(model.psi[j]):.6e}")

print("\n--- Pore transport decomposition [mol/(m^2 s)] ---")
for ion in model.ions:
    print(f"\nIon: {ion}")
    for s in model.seg:
        print(
            f"seg={s:2d}, "
            f"diff={value(model.diffusive_term[ion, s]):.6e}, "
            f"conv={value(model.convective_term[ion, s]):.6e}, "
            f"elec={value(model.electromigration_term[ion, s]):.6e}"
        )

print("\n--- Objective value ---")
print(f"obj = {value(model.obj):.6e}")