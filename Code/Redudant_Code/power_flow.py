"""
Optimal Power Flow (OPF) implementation for power system contract negotiation.
"""
import gurobipy as gp
from gurobipy import GRB
from utils import Expando

class OptimalPowerFlow:
    """
    Implements the Optimal Power Flow problem using Gurobi optimization.
    """
    def __init__(self, input_data):
        self.data = input_data
        self.variables = Expando()
        self.constraints = Expando()
        self.results = Expando()
        self._build_model()

    def _build_variables(self):
        """Build optimization variables for the OPF problem."""
        # Generator production variables
        self.variables.generator_production = {
            (g, t, s): self.model.addVar(
                lb=0,
                name=f'Electricity production of generator {g} at t {t} in scenario {s}'
            ) for g in self.data.GENERATORS 
              for t in self.data.TIME
              for s in self.data.SCENARIOS_L
        }
        
        # Voltage angle variables
        self.variables.voltage_angle = {
            (n, t, s): self.model.addVar(
                lb=-gp.GRB.INFINITY,
                ub=gp.GRB.INFINITY,
                name=f'Voltage angle at node {n} at time {t} in scenario {s}'
            ) for n in self.data.NODES 
              for t in self.data.TIME
              for s in self.data.SCENARIOS_L
        }

    def _build_constraints(self):
        """Build constraints for the OPF problem."""
        # Balance equation at each node
        self.constraints.balance_constraint = {
            (n, t, s): self.model.addLConstr(
                gp.quicksum(
                    self.variables.generator_production[g, t, s] * self.data.mapping_generators[n][g] 
                    for g in self.data.GENERATORS
                ) - gp.quicksum(
                    self.data.load_capacity[l][t, s] * self.data.mapping_loads[n][l] 
                    for l in self.data.LOADS
                ),
                gp.GRB.EQUAL,
                gp.quicksum(
                    self.data.bus_susceptance[n][m] * self.variables.voltage_angle[m, t, s] 
                    for m in self.data.NODES
                ),
                name=f'Balance equation at node {n} at time {t} in scenario {s}'
            ) for n in self.data.NODES 
              for t in self.data.TIME
              for s in self.data.SCENARIOS_L
        }

        # Max flow between nodes
        self.constraints.max_flow_constraint = {
            (n, m, t, s): self.model.addLConstr(
                self.data.bus_susceptance[n][m] * (
                    self.variables.voltage_angle[n, t, s] - self.variables.voltage_angle[m, t, s]
                ),
                gp.GRB.LESS_EQUAL,
                self.data.branch_capacity[n][m],
                name=f'Max flow between nodes {n} and {m} at time {t} in scenario {s}'
            ) for n in self.data.NODES 
              for m in [node for node in self.data.NODES if node != n]
              for t in self.data.TIME
              for s in self.data.SCENARIOS_L
        }

        # Generator capacity constraints
        self.constraints.capacity_constraints = {
            (g, t, s): self.model.addLConstr(
                self.variables.generator_production[g, t, s],
                gp.GRB.LESS_EQUAL,
                self.data.generator_capacity[g],
                name=f'Max production of generator {g} at time {t} in scenario {s}'
            ) for g in self.data.GENERATORS[:6]
              for t in self.data.TIME
              for s in self.data.SCENARIOS_L
        }

        # Base load generator G6 constraint
        self.constraints.base_load_Gen6 = {
            (t, s): self.model.addLConstr(
                self.variables.generator_production['G6', t, s],
                gp.GRB.EQUAL,
                self.data.generator_capacity['G6'],
                name=f'Baseload generator G6 at time {t} in scenario {s}'
            ) for t in self.data.TIME
              for s in self.data.SCENARIOS_L
        }

        # Slack bus voltage angle constraint
        self.constraints.slack_bus_constraint = {
            (t, s): self.model.addLConstr(
                self.variables.voltage_angle[self.data.slack_bus, t, s],
                gp.GRB.EQUAL,
                0,
                name=f'Slack bus voltage angle at time {t} in scenario {s}'
            ) for t in self.data.TIME
              for s in self.data.SCENARIOS_L
        }

    def _build_objective(self):
        """Build the objective function for the OPF problem."""
        objective = self.data.PROB * gp.quicksum(
            self.data.generator_cost_a[g] * self.variables.generator_production[g, t, s] +
            self.data.generator_cost_b[g] * (
                self.variables.generator_production[g, t, s] * 
                self.variables.generator_production[g, t, s]
            )
            for g in self.data.GENERATORS 
            for t in self.data.TIME 
            for s in self.data.SCENARIOS_L
        )
        self.model.setObjective(objective, gp.GRB.MINIMIZE)

    def _build_model(self):
        """Initialize and build the complete optimization model."""
        self.model = gp.Model(name='Economic dispatch')
        self.model.Params.OutputFlag = 0
        self.model.Params.TimeLimit = 500
        self._build_variables()
        self._build_constraints()
        self._build_objective()
        self.model.update()

    def _save_results(self):
        """Save optimization results."""
        # Save objective value
        self.results.objective_value = self.model.ObjVal
        
        # Save generator production values
        self.results.generator_production = {
            g: {
                (t, s): self.variables.generator_production[g, t, s].x
                for t in self.data.TIME 
                for s in self.data.SCENARIOS_L
            } for g in self.data.GENERATORS
        }

        # Save nodal prices (dual variables of balance constraints)
        self.results.price = {
            n: {
                (t, s): (1/self.data.PROB) * self.constraints.balance_constraint[n, t, s].Pi
                for t in self.data.TIME 
                for s in self.data.SCENARIOS_L
            } for n in self.data.NODES
        }

        # Save generator revenues
        self.results.generator_revenue = {
            g: {
                (t, s): sum(
                    (self.results.price[n][t, s] - self.data.generator_cost_a[g]) * 
                    self.results.generator_production[g][t, s] - 
                    self.data.generator_cost_b[g] * self.results.generator_production[g][t, s] *self.results.generator_production[g][t, s]
                    for n in self.data.NODES
                    if self.data.mapping_generators.loc[g, n] == 1
                )
                for t in self.data.TIME
                for s in self.data.SCENARIOS_L
            } for g in self.data.GENERATORS
        }

    def run(self):
        """Run the optimization model."""
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            self._save_results()
        else:
            raise RuntimeError(f"Optimization of {self.model.ModelName} was not successful")

    def display_results(self):
        """Display optimization results."""
        print("\n-------------------   RESULTS  -------------------")
        print("Optimal energy production cost:")
        print(self.results.objective_value)
        print("\nOptimal generator dispatches:")
        print(self.results.generator_production)
        print("\nPrices at optimality:")
        print(self.results.price)