import os
import warnings
import logging
import pandas as pd
import matplotlib.pyplot as plt
from oemof.solph import (Bus, EnergySystem, Flow, Model, create_time_index, processing, NonConvex)
from oemof.solph.components import (Sink, Source, Converter, GenericStorage)
from oemof.solph import EnergySystem
from oemof.solph import views
import oemof.solph as solph

# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# read the input data file
filename = r"STEP_3/inputs/data.csv"
data = pd.read_csv(filename, sep=";")

# --- Unit Conversion: kWh -> MWh ---
# The thermal demand in the CSV is in kWh, but other parameters (prices) are EUR/MWh.
# We convert demand to MWh to maintain consistency.
data["thermal_demand"] = data["thermal_demand"] / 1000

filename2 = r"STEP_3/inputs/irradiance_riga.csv"
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
    Source(label="natural_gas",outputs={gas_bus: Flow(variable_costs=(data["gas_price"] + data["co2_price"]))}) # Prices are EUR/MWh
                )

## Biomass source (Unlimited supply)
# Cost is fixed per unit of heat produced by CHP2, so we set source cost to 0 here.
energysystem.add(
    Source(label="biomass_source", outputs={biomass_bus: Flow(variable_costs=0)})
)

# Waste Heat Source (from data)
data["waste_heat"] = data["waste_heat"]

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

print(f"Peak Thermal Demand: {max_demand:.2f} MWh")

energysystem.add(
    Sink(
        label="thermal_demand",
        inputs={thermal_bus: Flow(
            nominal_value=max_demand,
            fix=demand_fix
        )},
    )
)

# **Thermal storage is NOT included (as per Step 3 requirement).**
# Shortage Source (to prevent infeasibility)
energysystem.add(
    Source(
        label="thermal_shortage",
        outputs={thermal_bus: Flow(variable_costs=1000000)} # High cost to discourage use
    )
)


# --- Step 3: Heat Producers (5 Total + Solar) ---

# Component Sizing Strategy:
# Peak Demand is approx 8-9 MWh (based on 8000+ kWh).
# CHP1 (Gas): ~50% of peak -> 4.5 MW_th
# CHP2 (Biomass): ~40% of peak -> 3.5 MW_th
# Gas Boiler: ~100% of peak (backup) -> 9 MW_th
# Heat Pump: ~20% of peak -> 1.8 MW_th
# Solar: ~10% contribution -> 1 MW_th (nominal)

# 1. Combined heat and power plant (CHP 1) - fuel 1 (Gas)
# NonConvex parameters added.
energysystem.add(
    Converter(
        label="chp_1", 
        inputs={gas_bus: Flow()},
        outputs={
            electrical_bus: Flow(variable_costs=5), # O&M cost
            thermal_bus: Flow(
                nominal_value=4.5, 
                nonconvex=NonConvex(
                    minimum_uptime=5, 
                    minimum_downtime=3, 
                    startup_costs=50, 
                    shutdown_costs=10,
                    initial_status=0
                )
            )
        },
        conversion_factors={electrical_bus: 0.421, thermal_bus: 0.474}
    )
)

# 2. Combined heat and power plant (CHP 2) - fuel 2 (Biomass)
# Uses unlimited biomass. Fixed cost per unit of heat.
energysystem.add(
    Converter(
        label="chp_2",
        inputs={biomass_bus: Flow()}, 
        outputs={
            electrical_bus: Flow(variable_costs=5),
            thermal_bus: Flow(
                nominal_value=3.5,
                variable_costs=20, # Fixed cost per unit of heat
                nonconvex=NonConvex(
                    minimum_uptime=10, 
                    minimum_downtime=5, 
                    startup_costs=100, 
                    shutdown_costs=20,
                    initial_status=0
                )
            ) 
        },
        conversion_factors={electrical_bus: 0.421, thermal_bus: 0.474}
    )
)

# 3. Gas Boiler 1
# Backup/Peaking unit.
energysystem.add(
    Converter(
        label="gas_boiler_1",
        inputs={gas_bus: Flow()},
        outputs={
            thermal_bus: Flow(
                nominal_value=9,
                nonconvex=NonConvex(
                    minimum_uptime=2, 
                    minimum_downtime=2, 
                    startup_costs=20, 
                    shutdown_costs=5,
                    initial_status=0
                )
            )
        },
        conversion_factors={thermal_bus: 0.95}
    )
)

# 4. Heat Pump (MODELED AS A CONVERTER)  - waste heat source
COP = 3.0
energysystem.add(
    Converter(
        label="heat_pump",
        inputs={
            electrical_bus: Flow(), 
            waste_heat_bus: Flow()
        }, 
        outputs={thermal_bus: Flow(nominal_value=1.8)},                     
        conversion_factors={
            thermal_bus: COP, 
            waste_heat_bus: COP - 1 
        }
    )
)

# 5. Solar Thermal Collector 
solar_thermal_conversion_factor = 0.5
energysystem.add(
    Converter(
        label="solar_collector",
        inputs={ambient_heat_bus: Flow()},
        outputs={thermal_bus: Flow(nominal_value=1.0)}, # Nominal value scaling
        conversion_factors={
            # The conversion factor is the time-series irradiance data
            thermal_bus: solar_thermal_conversion_factor * irradiance_data["DNI"]
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
