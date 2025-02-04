import numpy as np
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.pso import PSO  # Importing PSO
from pymoo.termination import get_termination
from pymoo.visualization.scatter import Scatter

class TemporalInterferenceOptimization(Problem):
    def __init__(self):
        super().__init__(
            n_var=4,  # Number of decision variables
            n_obj=1,  # Convert to single objective
            n_constr=0,
            xl=np.array([-10.0, -10.0, 0.0, 0.0]),
            xu=np.array([10.0, 10.0, 5.0, 5.0])
        )

    def _evaluate(self, X, out, *args, **kwargs):
        x1 = X[:, 0]
        x2 = X[:, 1]
        current1 = X[:, 2]
        current2 = X[:, 3]

        # Original objectives
        f1 = -(current1 * np.cos(x1) + current2 * np.sin(x2))  # Example function
        f2 = (current1 - current2)**2 + (x1 - x2)**2  # Example function

        # Combine the objectives with a weighted sum
        weight_f1 = 0.5
        weight_f2 = 0.5
        combined_objective = weight_f1 * f1 + weight_f2 * f2

        # Assign the single objective
        out["F"] = np.column_stack([combined_objective])

# Instantiate the optimization problem
problem = TemporalInterferenceOptimization()

# Set up the PSO algorithm for optimization
algorithm = PSO(
    pop_size=100,  # Population size (number of particles)
    w=0.7,         # Inertia weight
    c1=1.5,        # Cognitive parameter
    c2=1.5         # Social parameter
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

