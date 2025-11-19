import os
import warnings
import logging
import pandas as pd
import matplotlib.pyplot as plt
from oemof.solph import (Bus, EnergySystem, Flow, Model, create_time_index, processing)
from oemof.solph.components import (Sink, Source, Converter, GenericStorage)
from oemof.solph import views
import oemof.solph as solph

# Levelized Cost of Heat
def LCOH(invest_cost, operation_cost, heat_produced, revenue=0, i=0.05, n=20):
    pvf = ((1 + i) ** n - 1) / ((1 + i) ** n * i)
    return (invest_cost + pvf * (operation_cost - revenue)) / (pvf * heat_produced)
    
# Equivalent Periodic Cost
def epc(invest_cost, i=0.05, n=20):
    af = (i * (1 + i) ** n) / ((1 + i) ** n - 1)
    return invest_cost * af

# read the input data file
# Adjust path for script execution context
base_path = "/Users/jekabsjan/Desktop/MSc⚡️/YEAR 2/Practical Optimization of energy networks/PROJECT 2/DistrictHeating_project"
filename = os.path.join(base_path, "STEP_3/inputs/data.csv")
data = pd.read_csv(filename, sep=";")

# Fill NaNs just in case
data = data.fillna(0)

filename2 = os.path.join(base_path, "STEP_3/inputs/irradiance_riga.csv")
irradiance_data = pd.read_csv(filename2, sep=";")
irradiance_data = irradiance_data.fillna(0)

# specifying the solver
solver = "cbc"
solver_verbose = False

# Create energy system
datetimeindex = create_time_index(2022, number=len(data))
energysystem = EnergySystem(timeindex=datetimeindex, infer_last_interval=False)

# --- Step 2: Buses, Sources, Sinks (Single Thermal Bus) ---

# Buses
electrical_bus = Bus(label="electrical_bus")
thermal_bus = Bus(label="thermal_bus")
gas_bus = Bus(label="gas_bus")
biomass_bus = Bus(label="biomass_bus") # New Biomass Bus
waste_heat_bus = Bus(label="waste_heat_bus") # New Waste Heat Bus
ambient_heat_bus = Bus(label="ambient_heat_bus") # for solar collector

energysystem.add(electrical_bus, thermal_bus, gas_bus, biomass_bus, waste_heat_bus, ambient_heat_bus)

# Excess electricity sink
energysystem.add(
    Sink(
        label="excess_electricity",
        inputs={electrical_bus: Flow(variable_costs=data["electricity_price"] * -1)}
    )
)

# Gas source with cost
energysystem.add(
    Source(label="natural_gas",outputs={gas_bus: Flow(variable_costs=(data["gas_price"] + data["co2_price"]) / 1000)})
)

# Biomass source (Unlimited supply)
# Cost is fixed per unit of heat produced by CHP2, so we set source cost to 0 here.
energysystem.add(
    Source(label="biomass_source", outputs={biomass_bus: Flow(variable_costs=0)})
)

# Waste Heat Source (from data)
# Using max constraint based on data profile
max_waste_heat = data["waste_heat"].max()
if max_waste_heat > 0:
    waste_heat_profile = data["waste_heat"] / max_waste_heat
else:
    waste_heat_profile = [0] * len(data)
    max_waste_heat = 0

energysystem.add(
    Source(
        label="waste_heat_source", 
        outputs={waste_heat_bus: Flow(max=waste_heat_profile, nominal_value=max_waste_heat)}
    )
)

# Grid electricity source
energysystem.add(
    Source(label="electricity_grid", outputs={electrical_bus: Flow(variable_costs=data["electricity_price"])})
)

# Ambient Heat source (Input to Solar Collector)
energysystem.add(
    Source(label="ambient_heat", outputs={ambient_heat_bus: Flow(variable_costs=0)})
)

# Thermal demand sink
max_demand = data["thermal_demand"].max()
demand_fix = data["thermal_demand"] / max_demand

energysystem.add(
    Sink(
        label="thermal_demand",
        inputs={thermal_bus: Flow(
            nominal_value=353e3,
            fix=demand_fix
        )},
    )
)

# Shortage Source (to prevent infeasibility)
energysystem.add(
    Source(
        label="thermal_shortage",
        outputs={thermal_bus: Flow(variable_costs=10000)} # High cost to discourage use
    )
)

# --- Step 3: Heat Producers (5 Total + Solar) ---

# 1. Combined heat and power plant (CHP 1) - fuel 1 (Gas) - Unchanged
energysystem.add(
    Converter(
        label="chp_1", # Renamed to match summary labels
        inputs={gas_bus: Flow(nominal_value=475e3)},
        outputs={
            electrical_bus: Flow(variable_costs=5),
            thermal_bus: Flow()
        },
        conversion_factors={electrical_bus: 0.421, thermal_bus: 0.474}
    )
)

# 2. Combined heat and power plant (CHP 2) - fuel 2 (Biomass)
# Uses unlimited biomass. Fixed cost per unit of heat.
# Efficiency: Kept same as before (0.421 el, 0.474 th) as not specified.
# Fixed cost per unit of heat: Added variable_costs=20 (placeholder) to thermal_bus output.
energysystem.add(
    Converter(
        label="chp_2",
        inputs={biomass_bus: Flow(nominal_value=225e3)}, # Now uses biomass
        outputs={
            electrical_bus: Flow(variable_costs=5),
            thermal_bus: Flow(variable_costs=20) # Fixed cost per unit of heat
        },
        conversion_factors={electrical_bus: 0.421, thermal_bus: 0.474}
    )
)

# 3. Gas Boiler 1 - Unchanged
energysystem.add(
    Converter(
        label="gas_boiler_1",
        inputs={gas_bus: Flow(nominal_value=50e3)},
        outputs={thermal_bus: Flow()},
        conversion_factors={thermal_bus: 0.95}
    )
)

# 4. Heat Pump (MODELED AS A CONVERTER)  - waste heat source
COP = 3.0
# Inputs: Electricity and Waste Heat.
# Output: Heat.
# Relation: Heat = Elec * COP.
# Energy Balance: Heat = Elec + WasteHeat.
# WasteHeat = Heat - Elec = Heat - Heat/COP = Heat * (1 - 1/COP).
# So for 1 unit of Heat output:
# Elec input = 1/COP.
# WasteHeat input = 1 - 1/COP.
# With COP=3: Elec=1/3, Waste=2/3.
# Conversion factors relative to thermal_bus (output) being 1:
# electrical_bus: 1/COP
# waste_heat_bus: (COP-1)/COP
energysystem.add(
    Converter(
        label="heat_pump",
        inputs={
            electrical_bus: Flow(nominal_value=10e3), 
            waste_heat_bus: Flow()
        }, 
        outputs={thermal_bus: Flow()},                     
        conversion_factors={
            thermal_bus: COP, # Output is COP * Elec
            waste_heat_bus: COP - 1 # Input Waste is (COP-1) * Elec
        }
    )
)

# 5. Solar Thermal Collector - Unchanged
solar_thermal_conversion_factor = 0.5
energysystem.add(
    Converter(
        label="solar_collector",
        inputs={ambient_heat_bus: Flow()},
        outputs={thermal_bus: Flow()},
        conversion_factors={
            # The conversion factor is the time-series irradiance data
            thermal_bus: solar_thermal_conversion_factor * irradiance_data["GHI"]
        },
    )
)

# --- Step 4: Optimization and Results ---

print("Solving Optimization Model...")
model = Model(energysystem)

# Solve the optimization problem
try:
    model.solve(solver=solver, cmdline_options={"mipgap": 0.005}, solve_kwargs={"tee": solver_verbose})
    print("\nOptimization Complete.")
    
    # --- Results Summary ---
    print(f"Total Objective Value (Minimised System Costs): {model.objective():,.2f} EUR")
    results = processing.results(model)
    data_flow = processing.convert_keys_to_strings(results)

    print("\n### Heat Producer Output Summary (Total Flow to thermal_bus)")
    producer_labels = ["chp_1", "chp_2", "gas_boiler_1", "heat_pump", "solar_collector", "thermal_shortage"]
    summary_data = {}

    for label in producer_labels:
        # Correctly construct tuple key
        flow_key = (label, "thermal_bus")
        
        if flow_key in data_flow and 'sequences' in data_flow[flow_key] and 'flow' in data_flow[flow_key]['sequences']:
            flow_sum = data_flow[flow_key]['sequences']['flow'].sum()
            summary_data[label] = flow_sum
        else:
            summary_data[label] = 0.0 

    # Display results
    for producer, flow in sorted(summary_data.items(), key=lambda item: item[1], reverse=True):
        print(f"- {producer.replace('_', ' ').title():<18}: {flow:,.2f} MWh")

except Exception as e:
    print(f"\nOptimization Failed. Check the solver installation or data integrity. Error: {e}")
