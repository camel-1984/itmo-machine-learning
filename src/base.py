import math

s = "  hello    world"
result = " ".join(s.split())
print(result)  # "hello world"

# s.split() без аргументов сам:
# отбрасывает пробелы по краям,
# схлопывает любые последовательности пробелов/табов/переносов строк между словами,
# возвращает список слов.
# " ".join(...) склеивает их обратно через один пробел в строку

predictions = [0.9, 0.2, 0.7, 0.4, 0.85]
labels = [1, 0, 1, 1, 0]

def split_time(seconds: int) -> tuple[int, int, int]:
    h = seconds // 3600
    left = seconds % 3600
    m = left // 60
    s = left % 60
    return h, m, s

def normalize(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    if math.isclose(max_val, min_val): # float нельзя сравнить через ==
        return 0.0

    return (value - min_val) / (max_val - min_val)

def describe(data) -> str:
    if data is None: # null - синглтон, void* функция возвращает None
        return "нет данных"
    if not data:
        return "пусто"
    return f"элементов: {len(data)}"

# x is y — один ли это объект. Используется практически только с None, True, False.
# x == y — равны ли значения. Всё остальное сравнение. Аналог equals*
# isinstance(x, T) — принадлежит ли объект типу. Аналог instanceof*.

falsy = [
    False, None,                          # специальные
    0, 0.0, 0j,                           # числа (+ Decimal(0), Fraction(0))
    "", [], (), {}, set(), range(0),      # пустые коллекции
]

def accuracy(predictions: list[float], labels: list[int], threshold: float = 0.5) -> float:
    if not predictions:
        return 0.0
    correct = sum(int(p > threshold) == y for p, y in zip(predictions, labels))
    # sum - встроенная C функция быстрее python цикла
    return correct / len(predictions)


def report(predictions: list[float], labels: list[int], threshold: float = 0.5) -> None:
    for i, (p, y) in enumerate(zip(predictions, labels)):
        pred = int(p > threshold) # приведение bool к int (нативное)
        mark = "✓" if pred == y else "✗" # тернарник
        print(f"[{i}] pred={p:.2f} → {pred}, true={y}  {mark}")
    print(f"accuracy: {accuracy(predictions, labels, threshold):.2f}")

# c = 0 - переменная создана, значение 0
# c: int - переменная НЕ создана, только аннотация - значений по умолчанию нет
# c: int = 0 - создана, значение 0, плюс аннотация (не принято)

# Comprehensions - создание массива из уже существующего
squares = []
for x in range(5):
    squares.append(x * x)

squares = [x * x for x in range(5) if x > 5]

# columns = df.columns
# missing_columns = needed_columns - set(columns)
# сеты можно вычитать

# что делаем → откуда берём → фильтр.

# [x * 2 for x in data]       # list — квадратные
# {x * 2 for x in data}       # set  — фигурные, дубли уйдут
# (x * 2 for x in data)       # генератор — круглые

# запятая — это оператор, который создаёт tuple.

# генератор — ленивый объект, который считает
# элементы по одному по мере запроса и не хранит их в памяти.

# if any(v is None for v in row):
#     print("есть пропуски")

# Работают лениво: any останавливается на первом истинном,
# all — на первом ложном.

# sorted(words, key=len)  # по длине

# lambda — анонимная функция. lambda x: x * 2 эквивалентно def f(x): return x * 2.

if __name__ == "__main__":
    pass
# pass обязательная заглушка для проверки __name__

    h, m, s = split_time(3725)
    print(m)
    # print(f"{h}ч {m}м {s}с")

    # print(f"{normalize(0.5):.3f} вот так")
    # print(normalize(min_val=0.3, value=5))
    # print(f"{normalize(0.5, max_val=0.3)}")
# перегрузок нет - используется явная подстановка значений в параметры

    # report(predictions, labels)
