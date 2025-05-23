# This file contains the chemical constants like molar masses, densities, and concentrations of lactic acid and water.

M_h2o = 18.01528        # [g/mol] molar mass of water
M_lacticacid = 90.078   # [g/mol] molar mass of lactic acid 
rho_lacticacid = 1.209  # [g/mL] density of lactic acid at 25 °C
rho_h2o = 1.0           # [g/mL] density of water at 25 °C
c_p_h2o = 55.5          # [mol/L] concentration of pure water at 25 °C
c_p_lacticacid = rho_lacticacid / M_lacticacid * (10**3)  # [mol/L] concentration of pure lactic acid at 25 °C