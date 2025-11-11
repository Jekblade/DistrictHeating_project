import matplotlib.pyplot as plt
import dhnx
import pandas as pd
import oemof.solph
from pyomo.environ import SolverFactory

# Initialize thermal network
network = dhnx.network.ThermalNetwork()