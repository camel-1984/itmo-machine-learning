import pandas as pd
import numpy as np

# pandas объектно-ориентирован — почти всё метод.
df.sum()  df.dropna()  s.isna()  s.astype()  s.str.strip()
# Функций модуля мало, и они трёх видов: создание + объединение + pd.isna()
pd.read_csv, pd.to_numeric, pd.to_datetime, pd.DataFrame

# numpy функционален — почти всё np.*. (функция модуля)
np.isfinite(a)  np.log(a)  np.where(m)  np.diff(a)  np.concatenate([a, b])
# Методов у ndarray мало, и они про сам массив, а не про математику:
a.sum()  a.mean()  a.max()  a.reshape()  a.astype()  a.copy()  a.T  a.argmax()

# ============================================================
#  1. ИЕРАРХИЯ ТИПОВ
# ============================================================

# DataFrame = набор Series
# Series    = ndarray + индекс (метки строк) (ряд)
# ndarray   = плотный блок памяти одного типа (n dimensional)

df = pd.read_csv("data.csv")     # pandas.DataFrame
df["price"]                      # pandas.Series  — ОДИН столбец
df[["price", "area"]]            # DataFrame      — несколько (двойные скобки)
df["price"].to_numpy()           # numpy.ndarray  — массив

# np.array — функция (конструктор) copy=True по умолчанию
# np.ndarray — тип того, что она создаёт

a = np.array([1., 2., 3.])
len(a)      # 3
a.size      # 3
a.shape     # (3,)

m = np.array([[1, 2, 3],
              [4, 5, 6]])

len(m)      # 2 ← длина первой оси (число строк)
m.size      # 6 ← всего элементов
m.shape     # (2, 3)
m.ndim      # 2

# ============================================================
#  2. PANDAS: ЗАГРУЗКА И ОСМОТР
# ============================================================

df = pd.read_csv("data.csv")
df = pd.read_csv("data.csv", usecols=["price"])   # только нужные колонки

df.head()      # первые строки - df
df.shape       # (строк, столбцов) - tuple
df.columns     # название колонок - возвращает Index - хитрый массив с хеш таблицей

df.      # тип КАЖДОЙ колонки - series
# dtype определяется СОДЕРЖИМЫМ:
# одна ячейка "n/a" → вся колонка станет object (строки) → арифметика не работает

df.info()      # типы + сколько непустых - void
df.describe()  # статистика по числовым - df

dupl = df.duplicated(keep="first").sum()
# duplicated() - Series T/F
df.sum()          # Series по колонкам
df.sum().sum()    # одно число

axis=k — это ось, которая исчезает. Работает для любой размерности:
a.shape          # (2, 3, 4)
a.sum(axis=1)    # (2, 4)  ← тройка убралась
df_copy = df_copy[~blank.any(axis=1)].copy()
# убрать ось строк все строки переходят в столбец ~blank в любом из значений строки

df_copy[pos] = df_copy[pos].apply(lambda s: s.astype("string").str.strip()).replace("", pd.NA)
# Правило: если метод есть у DataFrame — вызывай напрямую, .apply только когда его нет.

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
filtered = df[df["price"] > 0].reset_index(drop=True)
df_copy = df_copy.sort_values(by=["date", "market"]).reset_index(drop=True)
# Список нужен только когда ключей несколько:
# вообще если колонок несколько то списком пихаем а не просто через запятую

.reset_index()          # после agg: ключи из индекса → в колонки - перейти к 0 1 2
.reset_index(drop=True) # после сортировки: удалить текущие индексы - перейти к 0 1 2

df_copy = df_copy.sort_values(by=["observations"] + group_by, ascending=[False] + [True] * len(group_by))
# иерархия сортировок
# [] - для конкатенации списков, * len ибо необходима равная длина group_by и [True] -
# чтобы показать что все элементы group by по возрастанию сортировать

# ============================================================
#  4. PANDAS: ОЧИСТКА
# ============================================================

df["price"].isna().sum()      # сколько пропусков
df.dropna(subset=["price"])   # удалить строки с NaN
# Удаление строк из DataFrame — операция над всей таблицей, её нельзя собрать поколоночно.
df["price"].fillna(df["price"].median())

df.loc[df["price"].isna(), "price"] = df["price"].median()
# менять исходный df можно только через явное .loc / .iloc
# все остальное вернет новый объект

prices = pd.to_numeric(df["price"], errors="coerce")
# сохраняет пришедший тип - есть дробные сделает все float64 и тд
# возвращает тот же контейнер, что и был на входе
# errors="raise" (по умолчанию) → ValueError на первом мусоре
# errors="coerce" → мусор становится NaN, не падает

pd.isna(a)
# nan None NaT NA что угодно
nan - numpy в pandas напечатет NaN принято nan
NA и NaT - pandas
pd.NA - не привязан к float заменяем на него когда string

np.nan   # float-колонки, numpy-массивы — классика
pd.NA    # nullable-типы: Int64, string, boolean
pd.NaT   # datetime
None     # object-колонки, Python-код

.str - строки
.cat - категориональные данные
.dt - дата время
# df_copy["year"] = df_copy["date"].dt.year
# df_copy["month"] = df_copy["date"].dt.month
# Это называется "accessor" (аксессор/накопитель методов).a
# Вне pd np не существует
# необходимо юзать с Series, чтобы выполнить операцию нужную
(date - df_copy["date"])          # Timedelta
(date - df_copy["date"]).dt.days  # int64

blank = market.str.strip() == ""
non_valid = market.isna() | blank
#.str.strip() убирает пробелы по краям возвращает series, дальше скалярное сравнение с пустой строкой.
astype(str) # нужен, чтобы .str не спотыкался, если в колонке смешаны типы.
# .str работает только с Series

parsed = pd.to_datetime(date, format="%Y-%m-%d", errors="coerce")
# преобразовывает object в datetime

# Когда в [] передаётся булев DataFrame а не Series, pandas делает не фильтрацию строк, а поэлементное маскирование:
# где True — значение остаётся, где False — на его место ставится NaN.

[col for col in needed if col not in df.columns]   # O(n × m)
needed - set(df.columns)                            # O(n + m)
# разность сетов быстрее генератора списка из-за hash table

for col in pos:
    df[col] = df[col].str.split().str.join(" ")
#эквивалентно
df[pos] = df[pos].apply(lambda s: s.str.split().str.join(" "))
# apply медленнее чем pd метод

df_valid["market"] = df_valid["market"].astype("string").str.split().str[:2].str.join(" ")
# взять 2 слова

f.groupby("market")["price"].mean()
# groupby - ленивый ничего не делает без терминальной функции (сортирует по умолчанию)
# возвращает Series чтобы вернуться к DF надо reset_index() и потом только сортировать
# Читается: сгруппируй по market → возьми price → посчитай среднее в каждой группе.
# market
# Herat 75.0
# Kabul 110.0

df.groupby(keys).agg(
    имя_результата=("исходная_колонка", "функция"),
    ...
)
# на выходе схлопнутый (сгруппированный) df где сначала колонки из keys потом колонки агрегации
# После группировки метками строк (Индекс) становятся сами значения ключей: -> надо reset без drop

report["price_range"] = report["max_price"] - report["min_price"]
# присобачить новую колонку в df

# ============================================================
#  PYTHON-ОПЕРАТОРЫ НА SERIES / DATAFRAME
# ============================================================
# ПРИНЦИП: скалярный контекст ломается на векторе.
# Всё, что должно свести Series к ОДНОМУ значению, либо падает,
# либо молча делает не то. Нужен вектор ответов → ищи метод pandas.

# ПИТОНОВСКОЕ        ЧТО ПРОИСХОДИТ                      АНАЛОГ PANDAS
# ------------------------------------------------------------------------
# x in s             ищет по ИНДЕКСУ, не по значениям    s.isin([...])
# if s:              ValueError: ambiguous               s.any() / s.all() / s.empty
# and / or / not     ValueError: ambiguous               &  |  ~   (+ скобки!)
# len(df)            работает — число строк              df.shape[0] явнее
# max(s), sum(s)     работают, но Python-цикл            s.max(), s.sum()
# sorted(s)          вернёт list, потеряет индекс        s.sort_values()
# str(s)             строка всего объекта                s.astype(str)
# round(s, 2)        работает через __round__            s.round(2)
# abs(s)             работает                            s.abs()
# reversed(s)        не то                               s[::-1]

# Широкая таблица
# Рынки превратились из значений колонки market в имена колонок. Одна строка = один месяц.
# Это называется pivot — разворот таблицы.
df_copy = df_copy.groupby(["year", "month", "market"])["price"].median().unstack("market")
# Берёт один уровень индекса и делает из его уникальных значений колонки.
df_copy = df_copy.rename_axis(columns=None) # market становится подписью оси, надо убрать
# по умолчанию None
rename_axis(index=None)              # стереть имя оси строк
rename_axis(columns="рынок")         # задать имя оси колонок
rename_axis(index=None, columns=None) # обе сразу
df.index.name      # имя оси СТРОК
df.columns.name    # имя оси КОЛОНОК
# Осей две — у любого DataFrame.

markets = ["fasf", "dafaf"]
df_copy = df_copy[markets] # поставить колонки в порядке списка markets

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

a = np.array([100., 110., 99.])
np.diff(a)          # array([ 10., -11.]) - вернет пустой ndarray если меньше 2 элементов
a[1:] - a[:-1]      # array([ 10., -11.])   ← идентично

prices = np.array([100., -5., 200.])
mask   = np.array([True, False, True])
prices[mask]        # [100., 200.] длина 2 — элемента нет
prices * mask       # [100., 0., 200.] длина 3 — на месте ноль

# ============================================================
#  7. NaN И БЕСКОНЕЧНОСТЬ
# ============================================================

# NaN = Not a Number, только для float (в int-массиве невозможен)
# type(np.nan) → float

# inf не приводится к Nan

np.nan == np.nan     # False ← не равен даже себе
np.nan > 0           # False
np.nan < 0           # False
# обычным сравнением не поймать

np.isnan(a)          # True где NaN
np.isinf(a)          # True где +inf или -inf
np.isfinite(a)       # True, где нормальное число ← используй это
# не работает со строками

a = a[np.isfinite(a)]                  # чистка одной строкой
np.array([1.0, 2.0, np.nan]).mean()    # nan — один NaN травит всю агрегацию
np.array([1.0]) / 0                    # inf (только warning, не исключение)

# ============================================================
#  8. ПРИВЕДЕНИЕ ТИПОВ
# ============================================================

np.asarray(x, dtype=float)   # примет список/кортеж/массив - репрезентирующий числовое значение;
# ValueError если не привести - пришел "aboba"
np.array # копирует всегда (по умолчанию copy=True)
np.asarray # копирует только если нужно
# если только читаем а потом применяем маску (получим новый объект) нет смысла использовать array и сразу создвать новый объект

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
df_copy = df.copy()
df_copy[pos] = ... # явная запись -> нужна копия

df = df[mask] # перепривязка ссылки (копия не нужна и так новый объект)
df_valid = df[mask] # чтобы не путаться и сохранять ссылку на исходный объект в функции

# у СПИСКА срез — копия, у ndarray — view.
