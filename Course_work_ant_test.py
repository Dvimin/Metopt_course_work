import random
from Course_work_ant import AntColonyAlgorithm
from Course_work_din import din_method
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from Course_work_gif import plot_graph_with_path

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
        for iteration in range(self.n_iterations):
            if iteration <= 10 or iteration >=90:
                print()
                print(f"---------------------> Итерация {iteration + 1}/{self.n_iterations} <------------------")
                ants = [Ant(random.choice(self.all_cities)) for _ in range(self.n_ants)]
                for ant_index, ant in enumerate(ants):
                    print(f" Муравей {ant_index + 1}/{self.n_ants} начинает маршрут в городе {ant.current_city}")
                    print("Шаг/Город", end=" ")
                    for i in range(len(self.all_cities)):
                        print(f"& {i}", end=" ")  # заголовки столбцов
                    print("\\\\ \\hline")
                    step = 0
                    while not self._satisfies_class_requirements(ant):
                        print(f"{step}", end=" ")
                        step += 1
                        next_city = self._select_next_city_print(ant)
                        ant.visit_city(next_city)

                        #print(f"  Муравей {ant_index + 1} посетил город {next_city}, маршрут: {ant.path}")

                    cost = ant.path_cost(self.time_matrix)
                    print(f" Муравей {ant_index + 1} завершил маршрут с общей стоимостью {cost}: {ant.path[:]}")

                    if cost < best_cost:
                        best_cost = cost
                        best_path = ant.path[:]
                        print(
                            f" !!! Новый лучший маршрут найден муравьем {ant_index + 1} с стоимостью {best_cost}: {best_path}")
                    print("--------------------------------------------------------------------------")

                print(" Распространение феромона...")
                self._spread_pheromone_print(ants, best_cost, best_path)
                print(" Испарение феромона...")
                self._evaporate_pheromone_print()
                print(f" Лучший найденный маршрут на итерации: {best_path}\n")
            else:
                ants = [Ant(random.choice(self.all_cities)) for _ in range(self.n_ants)]
                for ant_index, ant in enumerate(ants):
                    while not self._satisfies_class_requirements(ant):
                        next_city = self._select_next_city_not_print(ant)
                        ant.visit_city(next_city)

                    cost = ant.path_cost(self.time_matrix)

                    if cost < best_cost:
                        best_cost = cost
                        best_path = ant.path[:]

                self._spread_pheromone_not_print(ants, best_cost, best_path)
                self._evaporate_pheromone_not_print()
        return best_path, best_cost

    def _select_next_city_not_print(self, ant):
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
            raise ValueError("Sum of probabilities is zero. Check pheromone table and distances.")
        next_city = np.random.choice(self.all_cities, 1, p=move_prob)[0]
        return next_city

    def _select_next_city_print(self, ant):
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
            raise ValueError("Sum of probabilities is zero. Check pheromone table and distances.")

        tab_width = 8
        print("    ", end="")
        for i, prob in enumerate(move_prob):
            if prob > 0:
                # Отформатированное значение вероятности для LaTeX
                print(f"& {prob:.4f} ", end="".ljust(tab_width - len(f"{prob:.4f} ")))
            else:
                # Посещённые города помечаем числом 0, для LaTeX
                print("& 0 ", end="".ljust(tab_width - len("& 0 ")))
        # Завершение строки таблицы для LaTeX
        print("\\\\ \\hline")


        next_city = np.random.choice(self.all_cities, 1, p=move_prob)[0]
        return next_city

    def _spread_pheromone_not_print(self, ants, best_cost, best_path):
        for ant in ants:
            for i in range(len(ant.path) - 1):
                if ant.path == best_path:
                    delta_pheromone = 1.0 / best_cost
                    self.pheromone[ant.path[i]][ant.path[i + 1]] += delta_pheromone
                    self.pheromone[ant.path[i+1]][ant.path[i]] += delta_pheromone

    def _evaporate_pheromone_not_print(self):
        self.pheromone *= (1.0 - self.decay)

    def _spread_pheromone_print(self, ants, best_cost, best_path):

        print("\nМатрица феромонов до обновления:")
        self._print_matrix_feromon(self.pheromone)

        print()
        for ant in ants:
            for i in range(len(ant.path) - 1):
                if ant.path == best_path:
                    delta_pheromone = 1.0 / best_cost
                    self.pheromone[ant.path[i]][ant.path[i + 1]] += delta_pheromone
                    self.pheromone[ant.path[i + 1]][ant.path[i]] += delta_pheromone  # Убедитесь, что матрица симметрична
                    #print(f"Усиление феромона на ребре ({ant.path[i]}, {ant.path[i + 1]}): {delta_pheromone}")

        print("\nМатрица феромонов после обновления:")
        self._print_matrix_feromon(self.pheromone)

    def _evaporate_pheromone_print(self):
        for i in range(len(self.pheromone)):
            for j in range(len(self.pheromone[i])):
                if i != j:  # Игнорируем диагональные элементы
                    before_evaporation = self.pheromone[i][j]
                    self.pheromone[i][j] *= (1.0 - self.decay)
                    after_evaporation = self.pheromone[i][j]
                    #print(f"Испарение феромона на ребре ({i}, {j}): {before_evaporation} -> {after_evaporation}")
        print("\nМатрица феромонов после испарения:")
        self._print_matrix_feromon(self.pheromone)

        # for row in self.pheromone:
        #     print(" ".join(f"{pheromone:.4f}" for pheromone in row))


    def _print_matrix_feromon(self, pheromone):

        print("Город/Город", end=" ")
        for i in range(len(pheromone)):
            print(f"& {i}", end=" ")  # заголовки столбцов
        print("\\\\ \\hline")

        for i, row in enumerate(pheromone):
            pheromone[i][i] = 0.0  # Обнуляем диагональные элементы
            print(f"    {i}", end=" ")  # номер строки/города
            for value in row:
                print(f"& {value:.4f}", end=" ")  # значения феромонов
            print("\\\\ \\hline")  # завершаем строку

# Шаг/Город & 0 & 1 & 2 & 3 & 4 & 5 \\ \hline
#     1 & 0      & 0.1722 & 0.0766 & 0.0431 & 0.0191 & 0.6890 \\ \hline
#     2 & 0      & 0.1158 & 0.7235 & 0      & 0.0804 & 0.0804 \\ \hline
#     3 & 0      & 0.2975 & 0.6694 & 0      & 0.0331 & 0      \\ \hline
#     4 & 0      & 0.3378 & 0      & 0      & 0.6622 & 0      \\ \hline

# Веса
time_matrix = [
    [0, 2, 3, 4, 6, 1],
    [2, 0, 7, 5, 3, 3],
    [3, 7, 0, 2, 5, 2],
    [4, 5, 2, 0, 6, 6],
    [6, 3, 5, 6, 0, 9],
    [1, 3, 2, 6, 9, 0],
]

city_classes = [1, 2, 3, 1, 2, 3]
# 1: 'blue', 2: 'green', 3: 'red'
class_requirements = {1: 2, 2: 1, 3: 2}


# print("\nRunning Ant Colony Algorithm...")
# aco_algorithm = AntColonyAlgorithm(time_matrix, city_classes, class_requirements, n_ants=10, n_best=3,
#                                        n_iterations=100,
#                                        decay=0.5, alpha=1, beta=2)
#

aco_algorithm = AntColonyAlgorithm(time_matrix, city_classes, class_requirements, n_ants=50, n_best=3, n_iterations=20,
                                   decay=0.5, alpha=1, beta=2)


shortest_path, cost = aco_algorithm.run()
# shortest_path = [0, 5, 2, 3, 4]
# fig, axs = plt.subplots(1, 1, figsize=(6, 6))
# plot_graph_with_path(axs, time_matrix, shortest_path, city_classes, '')
shortest_path = [3, 2, 5, 0, 1]
fig, axs = plt.subplots(1, 1, figsize=(6, 6))
plot_graph_with_path(axs, time_matrix, shortest_path, city_classes, '')
plt.show()

print("Shortest path: ", shortest_path)
print("Cost of the path: ", cost)

#
# time_matrix = [
#     [0, 7, 9, 9, 1, 2, 6],
#     [7, 0, 4, 6, 6, 4, 7],
#     [9, 4, 0, 2, 9, 9, 2],
#     [9, 6, 2, 0, 2, 7, 10],
#     [1, 6, 9, 2, 0, 8, 8],
#     [2, 4, 9, 7, 8, 0, 4],
#     [6, 7, 2, 10, 8, 4, 0],
# ]
#
#
# city_classes = [3, 1, 2, 3, 1, 2, 1]
# shortest_path = [4, 0, 5, 2, 3]
# fig, axs = plt.subplots(1, 2, figsize=(16, 6))
# plot_graph_with_path(axs[1], time_matrix, shortest_path, city_classes, '')
# shortest_path = [0, 0]
# plot_graph_with_path(axs[0], time_matrix, shortest_path, city_classes, '')
#
#
#


