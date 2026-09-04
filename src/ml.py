from nis import cat

import numpy as np
import pandas as pd

# ============================================================
#  1. ИЕРАРХИЯ ТИПОВ
# ============================================================

# DataFrame = набор Series
# Series    = ndarray + индекс (метки строк)
# ndarray   = плотный блок памяти одного типа

df = pd.read_csv("data.csv")     # pandas.DataFrame
df["price"]                      # pandas.Series  — ОДИН столбец
df[["price", "area"]]            # DataFrame      — несколько (двойные скобки)
df["price"].to_numpy()           # numpy.ndarray  — массив

# np.array — функция (конструктор)
# np.ndarray — тип того, что она создаёт

# ============================================================
#  2. PANDAS: ЗАГРУЗКА И ОСМОТР
# ============================================================

df = pd.read_csv("data.csv")
df = pd.read_csv("data.csv", usecols=["price"])   # только нужные колонки

df.head()      # первые строки
df.shape       # (строк, столбцов)
df.columns     # название колонок

df.dtypes      # тип КАЖДОЙ колонки
# dtype определяется СОДЕРЖИМЫМ:
# одна ячейка "n/a" → вся колонка станет object (строки) → арифметика не работает

df.info()      # типы + сколько непустых
df.describe()  # статистика по числовым

dupl = df.duplicated(keep="first").sum()
# duplicated() - Series T/F

# ============================================================
#  3. PANDAS: ВЫБОРКА И ИНДЕКС
# ============================================================

df["price"]                       # колонка
df[df["price"] > 100]             # фильтр
df[(df["price"] > 100) & (df["area"] < 50)]   # & | ~ и СКОБКИ, как в numpy

df.loc[5, "price"]     # по МЕТКАМ (строка, столбец)
df.iloc[5, 0]          # по ПОЗИЦИЯМ (строка, столбец)
# базовая индексация int 0 -> n-1 может быть изменена

# Индекс = метки строк. После фильтрации становится дырявым: 0,3,7,12...
filtered = df[df["price"] > 0].reset_index(drop=True)   # починить нумерацию

# ============================================================
#  4. PANDAS: ОЧИСТКА
# ============================================================

df["price"].isna().sum()      # сколько пропусков
df.dropna(subset=["price"])   # удалить строки с NaN
df["price"].fillna(df["price"].median())

df.loc[df["price"].isna(), "price"] = df["price"].median()
# менять исходный df можно только через явное .loc / .iloc
# все остальное вернет новый объект

prices = pd.to_numeric(df["price"], errors="coerce")
# errors="raise" (по умолчанию) → ValueError на первом мусоре
# errors="coerce" → мусор становится NaN, не падает

pd.isna(a)
# Nan None NaT что угодно

.str - строки
.cat - категориональные данные
.dt - дата время
# это называется "accessor" (аксессор/накопитель методов).
# вне pd np не существует
# необходимо юзать с Series чтобы выполнить операцию нужную

blank = market.str.strip() == ""
non_valid = market.isna() | blank
#.str.strip() убирает пробелы по краям возвращает series, дальше скалярное сравнение с пустой строкой.
# astype(str) нужен, чтобы .str не спотыкался, если в колонке смешаны типы.
# .str работает только с Series

parsed = pd.to_datetime(date, format="%Y-%m-%d", errors="coerce")
# преобразовывает object в datetime

# Когда в [] передаётся булев DataFrame а не Series, pandas делает не фильтрацию строк, а поэлементное маскирование: г
# де True — значение остаётся, где False — на его место ставится NaN.


for col in pos:
    df[col] = df[col].str.split().str.join(" ")

df[pos] = df[pos].apply(lambda s: s.str.split().str.join(" "))

#эквивалентно

# ============================================================
#  5. ПЕРЕХОД PANDAS → NUMPY
# ============================================================

y = df["price"].to_numpy(dtype=float)
# пока данные с именами колонок — pandas; пошла математика — numpy

# ============================================================
#  6. NUMPY: БУЛЕВЫ МАСКИ
# ============================================================

prices = np.array([100.0, -5.0, 200.0, 0.0])

# наложение масок сдвигает индексы

mask = np.array([True, False, True])      # 1D
np.where(mask)          # (array([0, 2])), — кортеж из ОДНОГО массива
np.where(mask)[0]       # array([0, 2])      — сам массив

m = np.array([[True,  False, True],
              [False, True,  False]])

np.where(m)
# (array([0, 0, 1]), array([0, 2, 1])) в (0 0) (0 2) (1 1) строка столбец

np.flatnonzero(x) == np.whehre(valid)[0]

# fancy indexing - индексируешь массивом индексов
# скалярная индексация - индексируешь числом

prices > 0                # [True False True False] — массив булевых
prices[prices > 0]        # [100. 200.] — остаются, где True

mask = np.array([True, False, True, False])
prices[mask]              # [100. 200.]

#   ~ == not
#   & == and
#   | == or

prices[(prices > 0) & (prices < 150)]
# для массивов; скобки вокруг каждого условия ОБЯЗАТЕЛЬНЫ
# and/or/not на массиве → ValueError: truth value of an array is ambiguous

(prices > 0).sum()        # СКОЛЬКО (True считается за 1)
(prices > 0).mean()       # ДОЛЯ

# ============================================================
#  7. NaN И БЕСКОНЕЧНОСТЬ
# ============================================================

# NaN = Not a Number, только для float (в int-массиве невозможен)
# type(np.nan) → float

np.nan == np.nan     # False ← не равен даже себе
np.nan > 0           # False
np.nan < 0           # False
# обычным сравнением не поймать

np.isnan(a)          # True где NaN
np.isinf(a)          # True где +inf или -inf
np.isfinite(a)       # True где нормальное число ← используй это
# не работает со строками

a = a[np.isfinite(a)]                  # чистка одной строкой
np.array([1.0, 2.0, np.nan]).mean()    # nan — один NaN травит всю агрегацию
np.array([1.0]) / 0                    # inf (только warning, не исключение)

# ============================================================
#  8. ПРИВЕДЕНИЕ ТИПОВ
# ============================================================

np.asarray(x, dtype=float)   # примет список/кортеж/массив; ValueError если не привести
x.astype(float)              # метод у ndarray; ВСЕГДА копия

# float → float : NaN проходит спокойно (он и есть float)
# float → int : NaN → МУСОР, БЕЗ ошибки
np.array([1.0, np.nan]).astype(int)    # [1, -9223372036854775808]
# → чисти пропуски ДО приведения к int

prices = np.array([100.0, "abc", 200, "5.5"], dtype=object)
prices.dtype    # object
# без dtype=object numpy приведёт всё к одному типу — обычно к строкам

# ПОРЯДОК:
prices = np.asarray(prices, dtype=float)      # 1. в float
valid = np.isfinite(prices) & (prices > 0)    # 2. маска
prices = prices[valid]                        # 3. фильтр

# .sum() вернёт np.int64, а не питоновский int
# приводим к int (int)

# ============================================================
#  9. КОПИЯ ИЛИ VIEW  ← источник багов
# ============================================================

df.copy(deep=True)

a = np.arange(10)
b = a[2:5]        # СРЕЗ = VIEW, окно в те же данные
b[0] = 99
a[2]              # 99 — изменился ОРИГИНАЛ

b = a[2:5].copy() # явная копия

# view  : обычные срезы + transpose(), .reshape(), .view()
# df.to_numpy() и .values под CoW возвращают read-only view

# копия : булева маска, fancy indexing, арифметика, astype, to_numpy(), drop_na, drop_duplicates
# asarray может вернуть тот же объект, если тип уже подходит

# В современном pandas и numpy большинство операций возвращают новые объекты, а не изменяют исходные.
df_copy = df.copy()
# явная копия

# у СПИСКА срез — копия, у ndarray — view. Не путать.

# ============================================================
#  10. ЧАСТЫЕ ОШИБКИ
# ============================================================

# and/or/not на массиве              → &, |, ~
# (a > 0) & (a < 5) без скобок       → приоритет операторов
# isinstance(arr, float)             → arr.dtype
# a.sort() присвоили результат       → sort() возвращает None
# astype(int) при наличии NaN        → мусор молча
# сравнение NaN через ==             → всегда False, нужен isnan
# срез ndarray считать копией        → это view
# loc[0] после фильтрации            → KeyError, нужен reset_index
# df["a"] vs df[["a"]]               → Series vs DataFrame
# несовпадение shape                 → 80% всех ошибок, печатай .shape