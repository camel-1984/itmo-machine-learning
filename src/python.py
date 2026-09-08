import math

# Общий принцип: держись одной библиотеки в пределах операции.
# Раз данные в pandas и сравнение в pandas — парсинг тоже в pandas.

def f(a, *, c):
f(1, c=2)   #
f(1, 2)     # TypeError
# * — разделитель. Всё, что после неё, можно передавать только по имени.

def f(a, /, b):
f(1, b=2)
f(a=1, b=2)  # TypeError
# / - разделитель. Всё до него нельзя передавать по имени

result["is_valid"] = not result["missing_columns"] and all(
        result[k] == 0 for k in ("duplicate_rows", "invalid_prices", "missing_markets", "invalid_dates"))
        # missing_columns - список пустой
        # остальное int = 0

result = all(x == 0 for x in result.values() if type(x) == int)

if not missing_columns:      # «список пуст»
if len(missing_columns) == 0 # то же самое, не принято (PEP 8)

d = dict() # не принято объявлять так
d = {}
dict = "что-то"
x = dict()      # TypeError
y = {}          # работает всегда

list - []
dict - {}
tuple - ()
str - ""
set - исключение -> set()
# set не держит порядок вставки - при print или for порядок не гарантирован
list({3, 1, 2})     # порядок всё ещё произвольный, просто зафиксирован в этом объекте
sorted({3, 1, 2})   # [1, 2, 3] — детерминированный

sorted(x)
# возвращает новый list не меняет оригинал
# sorted(words, key=len)  # по длине
# key - принимает функция
x.sort()
# возвращает None меняет оригинал только с list

# приведения типов делается через list() set() int() str()... объявление через [] () {}...
# исключение - comprehension там тоже нужно [] ()

x = (5)    # int, скобки просто группировка
y = (5,)   # tuple из одного элемента
z = 5,     # тоже tuple!
# запятая — это оператор, который создаёт tuple.

_, relative_change = ...
# распаковка кортежа
# через [1] - считается плохо

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
def __eq__(self, other): return self.x == other.x # вызывается при проверке через ==
isinstance(x, T) # принадлежит ли объект типу с наследованием. Аналог instanceof*.
type(x) # строго узнать тип

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

# Comprehensions - создание массива из уже существующего - генератор коллекции
squares = []
for x in range(5):
    squares.append(x * x)

squares = [x * x for x in range(5) if x > 5]
# что делаем → откуда берём → фильтр.

# columns = df.columns
# missing_columns = needed_columns - set(columns)
# сеты можно вычитать

# [x * 2 for x in data]       # генератор списка list — квадратные
# {x * 2 for x in data}       # генератор коллекции set  — фигурные, дубли уйдут
# (x * 2 for x in data)       # генератор выражения — круглые
# и тд

# генератор — ленивый объект, который считает
# элементы по одному по мере запроса и не хранит их в памяти.

# if any(v is None for v in row):
#     print("есть пропуски")

# Работают лениво: any останавливается на первом истинном,
# all — на первом ложном.

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
