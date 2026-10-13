import math
import pickle
from functools import partial
from pathlib import Path

def careful_metric(y_true, y_pred):
    return fbeta(y_true, y_pred, beta=0.5)

careful_metric = lambda y_true, y_pred: fbeta(y_true, y_pred, beta=0.5)

# Когда ты пишешь model = cls() (то же самое, что Model()), Python делает два шага:
#
# 1. Model.__new__(Model) создаёт пустой объект. Именно __new__ возвращает экземпляр.
# 2. Model.__init__(obj) заполняет этот объект атрибутами (self.x = ...).
# Он ничего не возвращает, поэтому -> None. Если вернуть из __init__ что-то кроме None, будет TypeError.

with threadpool_limits(limits=1):
# ограничить потоки

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "model"
# временная директория которая удалится после блоку with

# ============================================================
# *args И **kwargs (**params)
# ============================================================

# В ОБЪЯВЛЕНИИ функции — собрать «сколько угодно» аргументов:
#   *args     — лишние позиционные → tuple
#   **kwargs  — лишние именованные → dict
# Имена args / kwargs — соглашение; важны только * и ** (**params тоже можно).
def f(a, *args, **kwargs):
    print(a, args, kwargs)

f(1, 2, 3, x=4, y=5)   # 1 (2, 3) {'x': 4, 'y': 5}

# Типичное применение — ПРОБРОСИТЬ аргументы дальше, не перечисляя их:
def train(model_cls, X, y, **params):
    model = model_cls(**params)    # ** здесь уже РАСПАКОВКА dict обратно в аргументы
    return model.fit(X, y)

train(RandomForestClassifier, X, y, n_estimators=300, max_depth=10)

# Порядок в сигнатуре строго такой:
def g(a, b=0, *args, c, **kwargs): ...
# обычные → со значением по умолчанию → *args → keyword-only (c) → **kwargs

# * и ** в ОБЪЯВЛЕНИИ — собирают, в ВЫЗОВЕ — распаковывают (см. раздел 1).

# ** внутри {} — это слияние словарей,
# ** в вызове — разворот в аргументы,
# ** в сигнатуре — сборка в словарь.

# округлить до знаков
round(x, n)

# вернуть первые k индексы если сортировка
k_indices = argsort(distances)[:k]

# вернуть индекс минимума
i = np.argmin(q)


# ============================================================
# 1. ОПЕРАТОРЫ И ВЫРАЖЕНИЯ
# ============================================================

# --- Морж-оператор := (walrus, Python 3.8+) ---
# Присваивает значение И возвращает его как результат выражения.
# Обычное присваивание (x = 5) — инструкция: ничего не возвращает
# и не может стоять внутри if, comprehension и т.п. := делает присваивание выражением.
#
if (value := elements[i]) > boundary:
    return value
#
# Уместен, когда одно и то же выражение и проверяется, и используется дальше:
# экономит повторное вычисление и лишнюю строку.
#
# 1. Чтение потока построчно
while (line := f.readline()):
    process(line)
#   Без моржа: line = f.readline() до цикла + ещё раз в конце тела.
#
# 2. Фильтр + переиспользование результата
results = [y for x in data if (y := expensive(x)) is not None]
#   Без моржа expensive(x) вызвалась бы дважды.
#
# 3. Сохранение промежуточного значения в условии
if (n := len(items)) > 10:
    print(f"слишком много: {n}")

# --- Тернарный оператор ---
значение_если_истина if условие else значение_если_ложь
x = clicks // shows if shows > 0 else 0

# --- Кортеж создаёт запятая, а не скобки ---
x = (5)    # int, скобки — просто группировка
y = (5,)   # tuple из одного элемента
z = 5,     # тоже tuple!
e = ()     # единственное исключение: пустой tuple — это именно скобки

# --- Распаковка * и ** в аргументы ---
# * — распаковывает последовательность в позиционные аргументы:
lat_range = (55.0, 57.0)
lat.between(*lat_range) == lat.between(55.0, 57.0)
out_of_region = ~missing & ~(lat.between(*lat_range) & lon.between(*lon_range))

# ** — распаковывает словарь в именованные аргументы:
f(**{"left": 1, "right": 2}) == f(left=1, right=2)

# --- lambda ---
# Анонимная функция в одно выражение:
lambda x: x * 2   # ≈ def f(x): return x * 2

# ============================================================
# 2. ФУНКЦИИ И ПАРАМЕТРЫ
# ============================================================

# --- Аннотация возвращаемого значения ---
# Функция возвращает ровно один объект. return a, b — это один tuple (a, b),
# поэтому и аннотация одна: tuple[...].
def check_id(pd_id_source_id: pd.DataFrame) -> tuple[list, dict]:
    ...

# --- * и / в сигнатуре ---
# * — всё ПОСЛЕ неё передаётся только по имени (keyword-only)
def f(a, *, c): ...
f(1, c=2)    # ок
f(1, 2)      # TypeError

# / — всё ДО него передаётся только по позиции (positional-only)
def f(a, /, b): ...
f(1, b=2)    # ок
f(a=1, b=2)  # TypeError

# --- Перегрузок нет ---
# Одна функция с одним именем. Вместо перегрузок — значения по умолчанию
# и именованные аргументы (см. normalize() ниже).

# --- Функция без return возвращает None ---
# Аналог void, но технически функция всё равно возвращает объект — None.


# ============================================================
# 2.1 FUNCTOOLS.PARTIAL
# ============================================================

# partial(функция, *аргументы) — новая функция, у которой часть аргументов
# уже «вшита». Остальные передаёшь при вызове.
def power(base, exp):
    return base ** exp

square = partial(power, exp=2)
square(5)        # 25  ==  power(5, exp=2)

# Зачем: когда функцию надо ОТДАТЬ куда-то, а оно вызовет её со своими аргументами
# (map, sorted(key=...), df.apply, callback). Там свои аргументы не допишешь.
list(map(partial(round, ndigits=2), [1.234, 5.678]))   # [1.23, 5.68]
df["price"].apply(partial(normalize, min_val=0, max_val=100))

# То же через lambda — работает:
square = lambda x: power(x, exp=2)
# partial короче, когда просто фиксируешь аргументы; lambda — когда нужна логика.

# Вшивать лучше ПО ИМЕНИ (exp=2): позиционные вшиваются слева направо,
# partial(power, 2) зафиксирует base=2, а не exp.

# ============================================================
# 3. ПЕРЕМЕННЫЕ И АННОТАЦИИ
# ============================================================

c = 0        # переменная создана, значение 0
c: int       # только аннотация, переменная НЕ создана (обращение → NameError)
c: int = 0   # создана, значение 0, плюс аннотация (нормальная практика)

# Аннотации не проверяются при запуске — это подсказка для IDE и линтеров.
lst_a: list[int] | range
func(range(1, 10))

# range — ленивый: range(1, 5) не хранит [1, 2, 3, 4] в памяти,
# а хранит start, stop, step и вычисляет значение по индексу по запросу.

# _имя — соглашение: «внутреннее», снаружи модуля/класса не использовать.
# Работает для функций, переменных, методов. Интерпретатор это не запрещает.

# _ — имя для ненужного значения:
_, relative_change = compute()   # распаковка кортежа, первое значение выбрасываем
relative_change = compute()[1]  # работает, но читается хуже


# ============================================================
# 4. COMPREHENSIONS И ГЕНЕРАТОРЫ
# ============================================================

# Comprehension — короткая запись «создать коллекцию из другой коллекции».
squares = []
for x in range(5):
    if x % 2 == 0:
        squares.append(x * x)

squares = [x * x for x in range(5) if x % 2 == 0]   # [0, 4, 16]
# что делаем → откуда берём → фильтр

# Вид скобок определяет результат:
[x * 2 for x in data]         # list comprehension
{x * 2 for x in data}         # set comprehension — дубли уйдут
{x: x * 2 for x in data}      # dict comprehension — есть двоеточие
(x * 2 for x in data)         # генераторное выражение — НЕ tuple!
tuple(x * 2 for x in data)    # если нужен именно tuple

# Пример dict comprehension:
props = properties(record)
row = {
    key: props.get(key)
    for key in (
        "id", "datetime", "region", "category", "severity", "light",
        "participants_count", "injured_count", "dead_count",
    )
}
props.get(key)  # вернёт None, если ключа нет
props[key]      # упадёт с KeyError

# Генератор — ленивый объект: считает элементы по одному по мере запроса
# и не хранит их все в памяти. Пройти по нему можно только один раз.

# any / all тоже ленивые: any останавливается на первом истинном, all — на первом ложном.
if any(v is None for v in row.values()):   # по dict без .values() идёт перебор КЛЮЧЕЙ
    print("есть пропуски")

# reversed(lst) возвращает не список, а итератор:
# return list(reversed(lst))


# ============================================================
# 5. КОЛЛЕКЦИИ
# ============================================================

# --- Создание пустых коллекций ---
d = {}         # принято так
d = dict()     # работает, но длиннее и медленнее

# Опасность: встроенные имена можно перезатереть
dict = "что-то"
x = dict()     # TypeError: 'str' object is not callable
y = {}         # литерал работает всегда
# Поэтому не называй переменные list, dict, str, id, sum и т.п.

# Пустые литералы:
[]      # list
{}      # dict
()      # tuple
""      # str
set()   # set ({} — это пустой dict, литерала пустого set нет)

# Литералы [] {} () — для создания, конструкторы list() set() int() str() — для приведения типов:
list("abc")  # ['a', 'b', 'c']
int("5")     # 5

# --- set не хранит порядок ---
# При print или for порядок не гарантирован.
list({3, 1, 2})     # порядок произвольный, просто зафиксирован в этом списке
sorted({3, 1, 2})   # [1, 2, 3] — детерминированный
# (dict, в отличие от set, порядок вставки хранит — с Python 3.7)

# --- Сортировка и разворот ---
sorted(x)                # НОВЫЙ list, оригинал не меняется; работает с любым итерируемым
sorted(words, key=len)   # сортировка по длине
x.sort()                 # меняет оригинал, возвращает None; только у list
lst.reverse()            # то же самое: in-place, None, только list
lst = lst.sort()         # типичная ошибка: lst становится None

# --- append vs extend ---
l.append(a)              # добавляет ОДИН элемент
l.append(a, b, c)        # TypeError: append() takes exactly one argument (3 given)
l.extend([a, b, c])      # добавляет все элементы итерируемого
result.extend(lst_a[i:])

# --- Хешируемость ---
# Ключи dict и элементы set должны быть хешируемыми.
# Грубо: хешируемое ≈ неизменяемое.
# Хешируемы:   int, float, str, None, bool, frozenset, tuple (если все его элементы хешируемы)
# Нехешируемы: list, dict, set
(1, [2])  # tuple, но нехешируем, т.к. внутри list

# --- Falsy-значения (в if дают False) ---
falsy = [
    False, None,                          # специальные
    0, 0.0, 0j,                           # нули (+ Decimal(0), Fraction(0))
    "", [], (), {}, set(), range(0),      # пустые коллекции
]

# Проверка на пустоту (PEP 8):
if not missing_columns:          # принято
if len(missing_columns) == 0:    # работает, но не принято


# ============================================================
# 6. СРАВНЕНИЕ И ТИПЫ
# ============================================================

x is y  # один ли это объект в памяти. На практике — почти только x is None.
#          (для bool пиши просто `if flag:`, а не `if flag is True:`)
x == y  # равны ли значения. Аналог equals() в Java.
#          Вызывает __eq__:
def __eq__(self, other): return self.x == other.x

isinstance(x, T)   # принадлежит ли объект типу С УЧЁТОМ наследования. Аналог instanceof
type(x)            # точный тип объекта, без учёта наследования
type(x) is T       # строгая проверка типа
# Нюанс: bool — наследник int, поэтому isinstance(True, int) → True, а type(True) is int → False

# --- Типы (классы) — обычные объекты first-class ---
# int, str, list и т.п. можно хранить в переменных/списках, сравнивать, вызывать как функцию (создать экземпляр)
order = [bool, int, float, complex]
order.index(int)   # 1 — сравнение классов по identity/equality работает как у любых объектов

max(order.index(t1), order.index(t2))
# max() по индексам, не по типам: сравнивает два int,
# а не сами классы — потом этим числом индексируют список обратно, чтобы получить тип.

# --- Общий тип для приведения двух значений ---
# числа: bool → int → float → complex (иерархия точности)
# последовательности: range → tuple → list (иерархия гибкости)
# если типы из разных семейств или один из них str — общий тип str, т.к. str(x) работает всегда


# ============================================================
# 7. СТРОКИ И СРЕЗЫ
# ============================================================

s = "  hello    world"
result = " ".join(s.split())
print(result)  # "hello world"

# s.split() без аргументов:
#   - отбрасывает пробелы по краям,
#   - схлопывает любые последовательности пробелов/табов/переносов,
#   - возвращает список слов.
# " ".join(...) склеивает список обратно через один пробел.

# Срезы — работают для list, tuple, str. Возвращают НОВЫЙ объект.
# lst[start:stop:step], stop не включается
lst[1:4]   # индексы 1, 2, 3
lst[:3]    # первые три
lst[3:]    # с индекса 3 до конца
lst[-3:]   # последние три
lst[:-1]   # все, кроме последнего
lst[::2]   # каждый второй с начала
lst[1::2]  # каждый второй, начиная с индекса 1
lst[::-1]  # реверс
lst[:]     # поверхностная копия всего списка


# ============================================================
# 8. PATHLIB
# ============================================================

path = Path("/Users/rostislavmolcanov/Documents/itmo/curse3/ml/conspects/data/wfp_food_prices_ken.csv")
path = root / "data" / "raw" / "file.geojson"   # / склеивает пути (вместо os.path.join)
name = path.name                                # "file.geojson"
parent = path.parent.parent                     # на два уровня вверх
text = path.read_text(encoding="utf-8")


# ============================================================
# 8.1 PICKLE
# ============================================================

# Сохраняет ЛЮБОЙ Python-объект в байты и восстанавливает обратно (сериализация).
# Типичное применение — сохранить обученную модель / тяжёлый результат.
# import pickle — наверху файла: импорты всегда в начале модуля.

class Model:
    def save(self, path: Path) -> None:
        with open(path, "wb") as model_file:   # "wb" — запись байтов
            pickle.dump(self, model_file)     # self — сохраняем сам объект

    @classmethod
    def load(cls, path: Path) -> "Model":
        with open(path, "rb") as model_file:   # "rb" — чтение байтов
            return pickle.load(model_file)

# --- Зачем with, а не просто присваивание ---
# Открытый файл — ресурс ОС. Его надо закрыть: иначе дескриптор висит,
# а записанные байты могут остаться в буфере и не дойти до диска.

# Просто присваивание:
model_file = open(path, "wb")
pickle.dump(self, model_file)   # упало здесь (исключение) →
model_file.close()              # до этой строки не дошли → файл не закрыт, .pkl может быть битым

# Честно без with пришлось бы писать так:
model_file = open(path, "wb")
try:
    pickle.dump(self, model_file)
finally:
    model_file.close()          # finally выполняется ВСЕГДА: и при ошибке, и при return

# with — это ровно этот try/finally, только короче: файл закроется
# при ЛЮБОМ выходе из блока — конец блока, исключение или return.
# Поэтому в load можно делать return прямо внутри with — файл всё равно закроется.

# as model_file — имя для того, что вернул open (сам файл).
# Работает не только с файлами, а с любым «контекстным менеджером»
# (объект с методами __enter__ / __exit__): соединения с БД, блокировки и т.п.

# Грабли:
# - pickle.load ВЫПОЛНЯЕТ код из файла → никогда не открывай чужие .pkl
# - только Python, и нужны те же классы/версии библиотек (sklearn другой версии может не загрузиться)
# - для таблиц лучше df.to_parquet, для конфигов и обмена — JSON


# ============================================================
# 8.2 JOBLIB
# ============================================================

# Обёртка над pickle для научных данных. Для sklearn-моделей — стандарт (так советует их документация).
# Файл сам открывает и закрывает → with не нужен.

joblib.dump(model, "model.joblib")
model = joblib.load("model.joblib")

joblib.dump(model, "model.joblib", compress=3)       # встроенное сжатие (0–9), у pickle нет
arr = joblib.load("big.joblib", mmap_mode="r")        # memory-mapping: массив читается с диска, не грузится весь в RAM

# Чем лучше pickle:
# - большие numpy-массивы (веса RandomForest, матрицы) пишет отдельно, без лишних копий → быстрее и меньше памяти
# - сжатие и mmap из коробки
# Для мелких объектов (dict, конфиг) выигрыша нет, pickle не хуже.

# Грабли те же, что у pickle (внутри тот же механизм):
# - load выполняет код → чужие файлы не открывать
# - привязка к версиям библиотек → фиксировать версии в requirements.txt
# В проде часто: ONNX (переносимый формат модели), skops (безопасный формат для sklearn),
# safetensors (веса нейросетей, PyTorch/HuggingFace).


# ============================================================
# 9. PANDAS / ДАННЫЕ
# ============================================================

# Общий принцип: держись одной библиотеки в пределах операции.
# Данные в pandas и сравнение в pandas → парсинг тоже в pandas.

# Множества можно вычитать:
missing_columns = needed_columns - set(df.columns)

result["is_valid"] = not result["missing_columns"] and all(
    result[k] == 0
    for k in ("duplicate_rows", "invalid_prices", "missing_markets", "invalid_dates")
)
not result["missing_columns"]  # список пропущенных колонок пуст
# остальные счётчики (int) равны 0

# То же, но по всем int-значениям словаря.
# type(x) is int, а не isinstance — чтобы не захватить bool.
is_valid = all(x == 0 for x in result.values() if type(x) is int)


# ============================================================
# 10. ASSERT И ИСКЛЮЧЕНИЯ
# ============================================================

assert условие, сообщение
assert clicks <= shows, "Clicks greater than shows"
# Для отладки, тестов и проверки внутренних инвариантов.
# При запуске python -O все assert отключаются →
# НЕ используй assert для проверки пользовательского ввода — там raise ValueError(...).

# Все исключения в Python — как unchecked в Java:
# их не нужно объявлять (throws нет) и не обязательно ловить.


# ============================================================
# 11. ПРИМЕРЫ ФУНКЦИЙ
# ============================================================

def split_time(seconds: int) -> tuple[int, int, int]:
    h = seconds // 3600
    left = seconds % 3600
    m = left // 60
    s = left % 60
    return h, m, s   # возвращается один tuple, при вызове его распаковывают: h, m, s = ...


def normalize(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    # float через == сравнивать ненадёжно из-за ошибок округления (0.1 + 0.2 != 0.3)
    if math.isclose(max_val, min_val):
        return 0.0
    return (value - min_val) / (max_val - min_val)


def describe(data) -> str:
    if data is None:   # None — синглтон (единственный объект), аналог null
        return "нет данных"
    if not data:       # пустая коллекция
        return "пусто"
    return f"элементов: {len(data)}"


predictions = [0.9, 0.2, 0.7, 0.4, 0.85]
labels = [1, 0, 1, 1, 0]


def accuracy(predictions: list[float], labels: list[int], threshold: float = 0.5) -> float:
    if not predictions:
        return 0.0
    # sum по bool: True считается как 1, False как 0
    # sum — встроенная функция на C, обычно быстрее ручного цикла с +=
    correct = sum(int(p > threshold) == y for p, y in zip(predictions, labels))
    return correct / len(predictions)


def report(predictions: list[float], labels: list[int], threshold: float = 0.5) -> None:
    for i, (p, y) in enumerate(zip(predictions, labels)):
        pred = int(p > threshold)          # bool → int: True → 1, False → 0
        mark = "✓" if pred == y else "✗"   # тернарник
        print(f"[{i}] pred={p:.2f} → {pred}, true={y}  {mark}")
    print(f"accuracy: {accuracy(predictions, labels, threshold):.2f}")


# ============================================================
# 12. ENTRY POINT
# ============================================================

# Код внутри выполняется только при прямом запуске файла, но не при импорте.
# pass нужен, только если блок иначе был бы пустым.
if __name__ == "__main__":
    h, m, s = split_time(3725)
    print(f"{h}ч {m}м {s}с")

    print(f"{normalize(0.5):.3f}")
    print(normalize(min_val=0.3, value=5))   # именованные аргументы — в любом порядке
    print(normalize(0.5, max_val=0.3))

    report(predictions, labels)
