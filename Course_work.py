import random
from Course_work_ant import AntColonyAlgorithm
from Course_work_din import din_method
import matplotlib.pyplot as plt
import networkx as nx


def plot_graph_with_path(ax, time_matrix, path, city_classes, method):
    G = nx.Graph()
    for i in range(len(time_matrix)):
        for j in range(i + 1, len(time_matrix)):
            if time_matrix[i][j] > 0:
                G.add_edge(i, j, weight=time_matrix[i][j])

    pos = nx.spring_layout(G)

    # Рисуем узлы с цветом в соответствии с их классом
    colors = ['blue' if city_classes[node] == 1 else 'green' if city_classes[node] == 2 else 'red' for node in
              G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=500, ax=ax)
    nx.draw_networkx_labels(G, pos, font_color='white', font_weight='bold', ax=ax)

    # Рисуем ребра
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), alpha=0.3, ax=ax)

    # Подсвечиваем путь
    edges_in_path = list(
        zip(path, list(path[1:]) + [path[0]]))  # Convert path[1:] to list and append path[0] as a list element
    nx.draw_networkx_edges(G, pos, edgelist=edges_in_path, edge_color='red', width=2, ax=ax)

    # Рисуем рёберные веса
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

    ax.set_title(f'{method} {len(time_matrix)}x{len(time_matrix)} Graph with highlighted path')
    ax.axis('off')


def generate_time_matrix(size):
    matrix = [[random.randint(1, 10) for _ in range(size)] for _ in range(size)]
    # Создаем симметричную матрицу
    for i in range(size):
        for j in range(i, size):
            matrix[i][j] = matrix[j][i] = random.randint(1, 10) if i != j else 0
    return matrix


def generate_city_classes_and_requirements(size, n_classes=3):
    # Убедимся, что для каждого класса есть достаточно городов
    min_cities_per_class = size // n_classes
    city_classes = []
    for i in range(1, n_classes + 1):
        city_classes += [i] * min_cities_per_class

    # Если размер матрицы не делится нацело на количество классов, добавим оставшиеся города
    remaining_cities = size % n_classes
    for i in range(1, remaining_cities + 1):
        city_classes.append(i)

    # Перемешаем классы городов для разнообразия
    random.shuffle(city_classes)

    # Учитывая переменную размерность городов на класс,
    # зададим требования так, чтобы не превысить количество городов в наименьшем классе
    class_counts = [city_classes.count(c) for c in range(1, n_classes + 1)]
    min_class_count = min(class_counts)
    class_requirements = {i: min_class_count for i in range(1, n_classes + 1)}

    return city_classes, class_requirements


sizes = [i for i in range(3, 10, 2)]

for size in sizes:
    time_matrix = generate_time_matrix(size)
    city_classes, class_requirements = generate_city_classes_and_requirements(size)

    print(f"Size: {size}x{size}")
    print("Time Matrix:")
    for row in time_matrix:
        print(row)
    print("City Classes:", city_classes)
    print("Class Requirements:", class_requirements)

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    print("\nRunning Ant Colony Algorithm...")
    aco_algorithm = AntColonyAlgorithm(time_matrix, city_classes, class_requirements, n_ants=10, n_best=3,
                                       n_iterations=100,
                                       decay=0.5, alpha=1, beta=2)

    shortest_path, cost = aco_algorithm.run()
    print("Shortest path: ", shortest_path)
    print("Cost of the path: ", cost)
    plot_graph_with_path(axs[0], time_matrix, shortest_path, city_classes, 'ANT')
    print("-------------------------------------")

    print("\nRunning Dynamic Programming Method...")
    din_result, din_route = din_method(time_matrix, city_classes, class_requirements)
    print("Shortest path: ", din_route)
    print("Cost of the path: ", din_result)
    plot_graph_with_path(axs[1], time_matrix, din_route, city_classes, 'DIN')

    plt.show()

    print("=====================================\n")
