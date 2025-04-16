from pyomo.environ import ConcreteModel, Var, Param, Constraint, Expression, SolverFactory, value
from math import log

# Create model
model = ConcreteModel()

# Define constants and parameters
L = 100e-9  # Pore length [m]
model.F = Param(initialize=96485)  # Faraday constant (C/mol)
model.R = Param(initialize=8.314)  # Universal gas constant (J/(mol·K))
model.T = Param(initialize=298)  # Absolute temperature [K]
model.r_p = Param(initialize=3.47e-9)  # Pore radius [m]
model.e = Param(initialize=1.602e-19)  # Elementary charge [C]
model.epsilon_0 = Param(initialize=8.854e-12)  # Vacuum permittivity [F/m]
model.pi = Param(initialize=3.1415925)  # Pi
model.area = Param(initialize=0.000804096)  # Membrane area [m²]
model.eta = Param(initialize=1e-3)  # Viscosity of water [Pa·s]
model.tau_L = Param(initialize=0.87)  # Membrane tortuosity
model.epsion = Param(initialize=0.17)  # Membrane porosity
model.sigma = Param(initialize=-2.2e-3)  # Membrane surface charge density [C/m²]

# Define pressure difference delta_P as a variable (to be updated during optimization)
model.delta_P = Var(initialize=5e6, bounds=(1.5e6, 5e6))  # Pressure difference [Pa], range: 1.5–5 MPa

# Define ions and their properties
ions = ['H+', 'Li+', 'Ni++', 'Co++', 'SO4--']
z_ions = {'H+': 1, 'Li+': 1, 'Ni++': 2, 'Co++': 2, 'SO4--': -2}  # Ionic valence
c_bulk = {
    'H+': 0.01e3, 'Li+': 0.110e3, 'Ni++': 0.099e3,
    'Co++': 0.0639e3, 'SO4--': 0.2229e3
}  # Feed concentration [mol/m³]
c_permeate = {
    'H+': 0.0099e3, 'Li+': 0.048e3, 'Ni++': 0.00036e3,
    'Co++': 0.000254e3, 'SO4--': 0.029064e3
}  # Permeate concentration [mol/m³]
D_ions = {
    'H+': 9.31e-9, 'Li+': 1.67e-9, 'Ni++': 1.32e-9,
    'Co++': 1.46e-9, 'SO4--': 1.06e-9
}  # Diffusion coefficients [m²/s]
r_ion = {
    'H+': 2.82e-10, 'Li+': 3.82e-10, 'Ni++': 4.04e-10,
    'Co++': 4.23e-10, 'SO4--': 3.97e-10
}  # Ionic radii [m]

# Assign ion data to model parameters
model.ions = ions
model.z = Param(model.ions, initialize=z_ions)  # Valence
model.c_bulk = Param(model.ions, initialize=c_bulk)  # Bulk concentration
model.c_permeate = Param(model.ions, initialize=c_permeate)  # Permeate concentration
model.r_i = Param(model.ions, initialize=r_ion)  # Ionic radius

# Initialize variables
model.dc_dx = Var(model.ions, bounds=(None, None), initialize=0)  # Concentration gradient [mol/m⁴]
model.dpsi_dx = Var(bounds=(None, None), initialize=0)  # Electric potential gradient [V/m]
model.solute_flux = Var(model.ions, initialize=0, bounds=(None, None))  # Solute flux [mol/m²·s]
model.Jv = Var(initialize=0.9e-05, bounds=(0, None))  # Water flux [m/s]

# 1. Calculate water flux (Jv)
# Define osmotic pressure difference (Δπ)
def delta_pi_rule(model):
    delta_pi_feed = sum(model.c_bulk[ion] for ion in model.ions)  # Concentration at membrane surface
    delta_pi_perm = sum(model.c_permeate[ion] for ion in model.ions)  # Permeate concentration
    return model.R * model.T * (delta_pi_feed - delta_pi_perm) * 1e6  # [Pa]

model.delta_pi = Expression(rule=delta_pi_rule)

def water_flux_rule(model):
    return model.Jv == (model.r_p**2 * model.delta_P) / (8 * (model.tau_L * L / model.epsion))

model.water_flux = Constraint(rule=water_flux_rule)

# 2. Calculate diffusion hindrance factor K_i,d and convection hindrance factor K_i,a
def hindrance_factor_diffusive_rule(model, ion):
    lambda_val = model.r_i[ion] / model.r_p
    if lambda_val <= 0.95:
        return  (
            1 + 9 / 8 * lambda_val * log(lambda_val)
            - 1.56034 * lambda_val
            + 0.528155 * lambda_val ** 2
            + 1.91521 * lambda_val ** 3
            - 2.81903 * lambda_val ** 4
            + 0.270788 * lambda_val ** 5
            + 1.10115 * lambda_val ** 6
            - 0.435933 * lambda_val ** 7
        ) / ((1 - lambda_val) ** 2)
    else:
        return 0.984 * ((1 - lambda_val) / lambda_val)**(5/2)

model.K_d = Expression(model.ions, rule=hindrance_factor_diffusive_rule)

def hindrance_factor_convective_rule(model, ion):
    lambda_val = model.r_i[ion] / model.r_p
    numerator = 1 + 3.867 * lambda_val - 1.907 * lambda_val**2 - 0.834 * lambda_val**3
    denominator = 1 + 1.867 * lambda_val
    return numerator / denominator

model.K_a = Expression(model.ions, rule=hindrance_factor_convective_rule)

# 3. Calculate electric potential gradient (dφ(x)/dx)
def electric_potential_gradient_rule(model):
    numerator = sum(
        model.z[ion] * (model.c_bulk[ion] - model.c_permeate[ion]) * model.Jv / (model.K_d[ion] * D_ions[ion]) 
        for ion in model.ions
    )
    denominator = sum(
        model.z[ion]**2 * model.c_bulk[ion] 
        for ion in model.ions
    )
    return model.dpsi_dx == numerator / (model.F / (model.R * model.T) * denominator)

model.electric_potential_gradient = Constraint(rule=electric_potential_gradient_rule)

# 4. Concentration gradient (dc_i/dx)
def concentration_gradient_rule(model, ion):
    return model.dc_dx[ion] == model.Jv / (model.K_d[ion] * D_ions[ion]) * (model.K_a[ion] * model.c_bulk[ion] - model.c_permeate[ion]) - (model.z[ion] * model.c_bulk[ion] * model.F / (model.R * model.T)) * model.dpsi_dx

model.concentration_gradient = Constraint(model.ions, rule=concentration_gradient_rule)

# 5. Calculate extended Nernst-Planck equation
def nernst_planck_rule(model, ion):
    return model.solute_flux[ion] == (
        -model.K_d[ion] * D_ions[ion] * model.dc_dx[ion] +
        model.K_a[ion] * model.c_bulk[ion] * model.Jv - 
        (model.z[ion] * model.c_bulk[ion] * model.K_d[ion] * D_ions[ion] * model.F / (model.R * model.T)) * model.dpsi_dx
    )

model.nernst_planck = Constraint(model.ions, rule=nernst_planck_rule)

# Invoke solver
solver = SolverFactory('ipopt')
solver.options['max_iter'] = 10000  # Increase maximum number of iterations
solver.options['tol'] = 1e-10       # Reduce tolerance

result = solver.solve(model, tee=True)

# Check results and output
if result.solver.termination_condition == 'optimal':
    print("Model solved successfully!")

    # Output water flux, electric potential gradient, concentration gradients, and ion fluxes
    Jv_value = value(model.Jv)
    print(f"Water flux (Jv): {Jv_value:.4e} m/s")

    dpsi_dx_value = value(model.dpsi_dx)
    print(f"Electric potential gradient (dφ/dx): {dpsi_dx_value:.4e} V/m")

    for ion in model.ions:
        dc_dx_value = value(model.dc_dx[ion])
        solute_flux_value = value(model.solute_flux[ion])
        print(f"Ion flux of {ion} (J_i): {solute_flux_value:.4e} mol/m²·s")
else:
    print("Solver failed to find an optimal solution.")

