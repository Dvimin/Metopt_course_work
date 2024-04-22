import itertools

def din_method(time_matrix, city_classes, class_requirements):
    n = len(time_matrix)  # Количество городов

    # Начальная точка
    start = 0


    # Создаем комбинации городов по классам
    city_combinations_per_class = {
        cls: [city for city in range(n) if city_classes[city] == cls]
        for cls in class_requirements
    }

    # Создаем все комбинированные маршруты с учетом классов и требований
    valid_routes = itertools.product(
        *(itertools.combinations(city_combinations_per_class[cls], class_requirements[cls])
          for cls in class_requirements)
    )

    min_route_cost = float('inf')
    min_route = []

    # Перебираем все валидные маршруты, проверяя их стоимость
    for route_combination in valid_routes:
        for route in itertools.permutations([city for cls_route in route_combination for city in cls_route]):
            # Стоимость маршрута с начальной и возвращающей точкой
            current_route_cost = sum(time_matrix[city][next_city] for city, next_city in
                                     zip((start,) + route, route + (start,)))
            if current_route_cost < min_route_cost:
                min_route_cost = current_route_cost
                min_route = route

    return min_route_cost, min_route
