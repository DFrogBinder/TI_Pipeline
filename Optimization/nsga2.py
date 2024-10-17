import numpy as np
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.operators.sampling.rnd import IntegerRandomSampling  # Updated import
from pymoo.operators.crossover.sbx import SBX  # Updated import
from pymoo.operators.mutation.pm import PM  # Updated import
from pymoo.visualization.scatter import Scatter

# Define the TI optimization problem
class TemporalInterferenceOptimization(Problem):
    def __init__(self):
        super().__init__(
            n_var=4,  # Number of decision variables (e.g., x1, x2, current1, current2)
            n_obj=2,  # Number of objectives (e.g., target intensity, minimize spread)
            n_constr=0,  # No constraints for now (add if needed)
            xl=np.array([-10.0, -10.0, 0.0, 0.0]),  # Lower bounds for variables
            xu=np.array([10.0, 10.0, 5.0, 5.0])  # Upper bounds for variables
        )

    def _evaluate(self, X, out, *args, **kwargs):
        # Example objectives based on the decision variables
        # X[:, 0] and X[:, 1] could represent electrode positions
        # X[:, 2] and X[:, 3] could represent currents

        x1 = X[:, 0]
        x2 = X[:, 1]
        current1 = X[:, 2]
        current2 = X[:, 3]

        # Objective 1: Maximize stimulation intensity at a target region
        f1 = -(current1 * np.cos(x1) + current2 * np.sin(x2))  # Example function

        # Objective 2: Minimize the spread of stimulation
        f2 = (current1 - current2)**2 + (x1 - x2)**2  # Example function

        # Assign objectives
        out["F"] = np.column_stack([f1, f2])

# Instantiate the optimization problem
problem = TemporalInterferenceOptimization()

# Set up the NSGA-II algorithm for multi-objective optimization
algorithm = NSGA2(
    pop_size=100,
    sampling=IntegerRandomSampling(),  # Updated sampling method
    crossover=SBX(prob=0.9, eta=15),  # Updated crossover method
    mutation=PM(eta=20),  # Updated mutation method
    eliminate_duplicates=True
)

# Define the termination criterion
termination = get_termination("n_gen", 5000)

# Perform the optimization
result = minimize(
    problem,
    algorithm,
    termination,
    seed=1,
    save_history=True,
    verbose=True
)

# Plot the Pareto front (set of optimal solutions)
plot = Scatter()
plot.add(result.F, facecolor="red")
plot.show()

# Print the optimal solutions
print("Optimal solutions (Pareto Front):")
print(result.X)
