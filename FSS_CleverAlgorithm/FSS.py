import numpy as np
import matplotlib.pyplot as plt

# Fish School Search Algorithm
class FishSchoolSearch:
    def __init__(self, n_fish, dimensions, fitness_function, iterations, step_individual=1, step_volitive=0.2):
        self.n_fish = n_fish
        self.dimensions = dimensions
        self.fitness_function = fitness_function
        self.iterations = iterations
        self.step_individual = step_individual
        self.step_volitive = step_volitive

        # Initialize fish positions randomly in a [-10, 10] space
        self.positions = np.random.uniform(-10, 10, (n_fish, dimensions))
        self.weights = np.ones(n_fish)
        self.best_solution = None
        self.best_fitness = np.inf  # We're minimizing, so start with infinity

    def optimize(self):
        for iteration in range(self.iterations):
            fitness = np.apply_along_axis(self.fitness_function, 1, self.positions)
            
            # Update best solution
            min_fitness_index = np.argmin(fitness)
            if fitness[min_fitness_index] < self.best_fitness:
                self.best_fitness = fitness[min_fitness_index]
                self.best_solution = self.positions[min_fitness_index]

            # Individual movement
            self.individual_movement(fitness)

            # Update weights
            self.update_weights(fitness)

            # Collective movement
            self.collective_movement()

            # Reduce step size over time
            self.step_individual *= 0.99
            self.step_volitive *= 0.99

        return self.best_solution, self.best_fitness

    def individual_movement(self, fitness):
        for i in range(self.n_fish):
            random_direction = np.random.uniform(-1, 1, self.dimensions)
            new_position = self.positions[i] + self.step_individual * random_direction
            new_fitness = self.fitness_function(new_position)
            if new_fitness < fitness[i]:  # Accept if fitness improves
                self.positions[i] = new_position

    def update_weights(self, fitness):
        fitness_improvement = np.maximum(0, np.min(fitness) - fitness)
        self.weights += fitness_improvement
        self.weights = np.maximum(1, self.weights / np.max(self.weights))  # Normalize weights

    def collective_movement(self):
        barycenter = np.average(self.positions, axis=0, weights=self.weights)
        for i in range(self.n_fish):
            direction_to_barycenter = barycenter - self.positions[i]
            self.positions[i] += self.step_volitive * direction_to_barycenter

# Define the landscape (fitness function)
def landscape_function(x):
    return x[0]**2 + x[1]**2 + 10 * np.sin(x[0]) + 7 * np.cos(x[1])

# Visualize the landscape
x = np.linspace(-10, 10, 400)
y = np.linspace(-10, 10, 400)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2 + 10 * np.sin(X) + 7 * np.cos(Y)

plt.figure(figsize=(10, 7))
plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(label='Fitness Value')
plt.title("Landscape: f(x, y) = x^2 + y^2 + 10sin(x) + 7cos(y)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Run Fish School Search
fss = FishSchoolSearch(n_fish=30, dimensions=2, fitness_function=landscape_function, iterations=100)
best_position, best_fitness = fss.optimize()

print("\nBest Position Found:", best_position)
print("Best Fitness Achieved:", best_fitness)
