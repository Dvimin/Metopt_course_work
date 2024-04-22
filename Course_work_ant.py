import numpy as np
import random

# Веса
matrix = [
    [0, 2, 3, 4, 6, 2],
    [2, 0, 7, 5, 3, 5],
    [3, 7, 0, 2, 5, 7],
    [4, 5, 2, 0, 6, 1],
    [6, 3, 5, 6, 0, 2],
    [1, 3, 2, 6, 9, 0],
]

# Классы городов: каждый класс своим цветом
city_classes = [1, 2, 3, 1, 2, 3]
# 1: 'blue', 2: 'green', 3: 'red'
color_map = {1: 'blue', 2: 'green', 3: 'red'}

class Ant:
    def __init__(self, start):
        self.path = [start]
        self.visited = {start}
        self.current_city = start

    def visit_city(self, city):
        self.path.append(city)
        self.visited.add(city)
        self.current_city = city

    def path_cost(self, time_matrix):
        cost = 0
        for i in range(len(self.path) - 1):
            cost += time_matrix[self.path[i]][self.path[i + 1]]
        cost += time_matrix[self.path[-1]][self.path[0]]
        return cost


class AntColonyAlgorithm:

    def __init__(self, time_matrix, city_classes, class_requirements, n_ants, n_best, n_iterations, decay, alpha=1,
                 beta=1, start=None):
        self.time_matrix = time_matrix
        self.city_classes = city_classes
        self.class_requirements = class_requirements
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.pheromone = np.ones((len(time_matrix), len(time_matrix))) / len(time_matrix)
        self.all_cities = list(range(len(time_matrix)))
        # self.start = start
    def _satisfies_class_requirements(self, ant):
        class_counts = {city_class: 0 for city_class in self.class_requirements.keys()}
        for city in ant.visited:
            city_class = self.city_classes[city]
            if city_class in class_counts:
                class_counts[city_class] += 1
        return all(class_counts[city_class] >= self.class_requirements[city_class] for city_class in class_counts)

    def run(self):
        best_cost = float('inf')
        best_path = []
        # if self.start is None:
        #     self.start = int(input(f"Введите начальную точку от 0 до {len(self.time_matrix) - 1}: "))

        for _ in range(self.n_iterations):
            # ants = [Ant(self.start) for _ in range(self.n_ants)]
            ants = [Ant(random.choice(self.all_cities)) for _ in range(self.n_ants)]
            for ant in ants:
                while not self._satisfies_class_requirements(ant):
                    next_city = self._select_next_city(ant)
                    ant.visit_city(next_city)
                cost = ant.path_cost(self.time_matrix)
                if cost < best_cost:
                    best_cost = cost
                    best_path = ant.path[:]
            self._spread_pheromone(ants, best_cost, best_path)
            self._evaporate_pheromone()

        return best_path, best_cost

    def _select_next_city(self, ant):
        pheromone = np.array([self.pheromone[ant.current_city][i]
                              if i not in ant.visited else 0
                              for i in self.all_cities])
        dist_inv = np.array([1 / self.time_matrix[ant.current_city][i]
                             if i not in ant.visited and self.time_matrix[ant.current_city][i] > 0 else 0
                             for i in self.all_cities])
        move_prob = (pheromone ** self.alpha) * (dist_inv ** self.beta)

        sum_prob = np.sum(move_prob)
        if sum_prob > 0:
            move_prob /= sum_prob
        else:
            # В случае, если все вероятности равны нулю, бросаем исключение, что-то не так с данными
            raise ValueError("Sum of probabilities is zero. Check pheromone table and distances.")

        # Avoid the case where all move probabilities are zero by adding a tiny probability to go to an unvisited city
        if not np.any(move_prob):
            unvisited = np.array([i for i in self.all_cities if i not in ant.visited])
            move_prob[unvisited] = 1.0 / len(unvisited)
            move_prob /= move_prob.sum()

        # Выбираем следующий город на основе расчетных вероятностей
        next_city = np.random.choice(self.all_cities, 1, p=move_prob)[0]
        return next_city


    def _spread_pheromone(self, ants, best_cost, best_path):
        for ant in ants:
            for i in range(len(ant.path) - 1):
                if ant.path == best_path:
                    self.pheromone[ant.path[i]][ant.path[i + 1]] += 1.0 / best_cost

    def _evaporate_pheromone(self):
        self.pheromone *= (1.0 - self.decay)
