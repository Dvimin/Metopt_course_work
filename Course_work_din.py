import itertools

time_matrix = [
    [0, 2, 3, 4, 6, 2],
    [2, 0, 7, 5, 3, 5],
    [3, 7, 0, 2, 5, 7],
    [4, 5, 2, 0, 6, 1],
    [6, 3, 5, 6, 0, 2],
    [1, 3, 2, 6, 9, 0],
]

city_classes = [1, 2, 3, 1, 2, 3]
class_requirements = {1: 2, 2: 2, 3: 2}


def din_method(time_matrix, city_classes, class_requirements):
    cities_to_visit = []
    for cls, num in class_requirements.items():
        cities_of_class = [i for i, c in enumerate(city_classes) if c == cls][:num]
        cities_to_visit.extend(cities_of_class)

    all_routes = itertools.permutations(cities_to_visit)

    min_route_cost = float('inf')
    min_route = ()

    for route in all_routes:
        class_counter = {cls: 0 for cls in class_requirements}
        for city in route:
            class_counter[city_classes[city]] += 1
        if any(class_counter[cls] < num for cls, num in class_requirements.items()):
            continue
        current_route_cost = time_matrix[0][route[0]] + sum(
            time_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1)) + time_matrix[route[-1]][0]
        if current_route_cost < min_route_cost:
            min_route_cost = current_route_cost
            min_route = route

    print(f"Минимальная стоимость маршрута: {min_route_cost}")
    print(f"Минимальный маршрут: {min_route}")

