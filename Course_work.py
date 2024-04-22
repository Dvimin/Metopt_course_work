from Course_work_ant import AntColonyAlgorithm
from Course_work_din import din_method

time_matrix = [
    [0, 2, 3, 4, 6, 2],
    [2, 0, 7, 5, 3, 5],
    [3, 7, 0, 2, 5, 7],
    [4, 5, 2, 0, 6, 1],
    [6, 3, 5, 6, 0, 2],
    [1, 3, 2, 6, 9, 0],
]

city_classes = [1, 2, 3, 1, 2, 3]
# 1: 'blue', 2: 'green', 3: 'red'
class_requirements = {1: 2, 2: 2, 3: 2}

aco_algorithm = AntColonyAlgorithm(time_matrix, city_classes, class_requirements, n_ants=10, n_best=3, n_iterations=100,
                                   decay=0.5, alpha=1, beta=2, start=0)

shortest_path, cost = aco_algorithm.run()
print("Shortest path: ", shortest_path)
print("Cost of the path: ", cost)


print('-------------------------------------')

din_method(time_matrix, city_classes, class_requirements)
