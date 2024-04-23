import itertools

# Веса
time_matrix = [
    [0, 2, 3, 4, 6, 1],
    [2, 0, 7, 5, 3, 3],
    [3, 7, 0, 2, 5, 2],
    [4, 5, 2, 0, 6, 6],
    [6, 3, 5, 6, 0, 9],
    [1, 3, 2, 6, 9, 0],
]

# Классы городов: каждый класс своим цветом
city_classes = [1, 2, 3, 1, 2, 3]

# Требования: сколько городов каждого класса мы должны посетить
class_requirements = {1: 2, 2: 2, 3: 2}

def din_method(time_matrix, city_classes, class_requirements):
    n = len(time_matrix)  # Количество городов

    # Начальная точка
    start = 0

    # Создаем комбинации городов по классам
    city_combinations_per_class = {
        cls: [city for city in range(n) if city_classes[city] == cls]
        for cls in class_requirements
    }

    print(f"Комбинации городов по классам: {city_combinations_per_class}")

    # Создаем все комбинированные маршруты с учетом классов и требований
    valid_routes = itertools.product(
        *(itertools.combinations(city_combinations_per_class[cls], class_requirements[cls])
          for cls in class_requirements)
    )

    min_route_cost = float('inf')
    min_route = []

    # Перебираем все валидные маршруты, проверяя их стоимость
    for route_combination in valid_routes:
        print(f"\nРассматриваем комбинацию: {route_combination}")
        for route in itertools.permutations([city for cls_route in route_combination for city in cls_route]):
            # Стоимость маршрута с начальной и возвращающей точкой
            current_route_cost = sum(time_matrix[city][next_city] for city, next_city in
                                     zip((start,) + route, route + (start,)))
            print(f"Проверяем маршрут: {((start,) + route, route + (start,))} со стоимостью {current_route_cost}")
            if current_route_cost < min_route_cost:
                min_route_cost = current_route_cost
                min_route = route
                print(f"Найден более дешевый маршрут: {min_route} со стоимостью {min_route_cost}")

    return min_route_cost, min_route

# Выводим результат работы функции с учетом классов и требований
result_cost, result_route = din_method(matrix, city_classes, class_requirements)
print(f"\nРезультат динамического метода: стоимость {result_cost}, маршрут {result_route}")