import pandas as pd
import numpy as np

p = np.clip(p, eps, 1 - eps)
# - если p < eps, берём eps;
# - если p > 1 - eps, берём 1 - eps;
# - иначе p остаётся как есть.


if not (np.isfinite(y_pred_nd).all() and
                (y_pred_nd >= 0).all() and
                (y_pred_nd <= 1).all()):

# all сводит массив TF к одному bool

is_counting = (X["algorithm"] == "counting_sort").to_numpy()
size = np.where(is_counting, X["n"] + X["value_range"], X["n"])
# условие, если true, если false

# Estimator — любой объект sklearn, который учится на данных.

X = X @ self.W_
# @ здесь и есть проекция. Матричное умножение это скалярные произведения, посчитанные сразу для всех объектов.

# --- numpy: маска vs массив номеров ---
a = np.array([10, 20, 30])

a[np.array([True, False, True])]   # [10, 30] — МАСКА: оставить/выкинуть; длина = len(a)
a[np.array([2, 0, 0, 1])]          # [30, 10, 10, 20] — НОМЕРА: какие позиции взять
# номера: любой порядок, повторы, любая длина

a[np.array([[0, 2],
            [1, 1]])]              # [[10, 30], [20, 20]] — форма результата = форма индекса

# --- pandas: булев DataFrame в [] — маскирование, НЕ фильтр ---
df = pd.DataFrame({"a": [1, 5], "b": [4, 2]})
df[df > 2]              # форма та же, строки не пропадают, где False → NaN
#      a    b
# 0  NaN  4.0
# 1  5.0  NaN
# (int стал float — из-за NaN)
df.where(df > 2, 0)     # то же, но False → 0 вместо NaN
df.mask(df > 2)         # наоборот: NaN там, где True

s[a,b]
df[a,b,c]
df[s] = df[a,b]

# --- Series / DataFrame → ndarray ---
s.to_numpy()               # ✅ принято: явно «дай массив»; не копирует, если можно
s.to_numpy(dtype=float)    #    можно сразу привести тип
np.array(s)                # ⚠️ работает, но ВСЕГДА копия и намерение читается хуже
s.values                   # ❌ устарело: для некоторых типов вернёт НЕ ndarray

features = df.drop(columns="target", errors="ignore")

# массив длина кол-во строк в X из valude
def predict(self, X: pd.DataFrame) -> np.ndarray:
    return np.full(len(X), self.value)

df.corrwith(df["target"], method="spearman") # series

mi = mutual_info_classif(features, df["target"], random_state=0) # ndarray

# |a - b| <= atol + rtol * |b|
# По умолчанию rtol=1e-5 и atol=1e-8,

std_maks = (np.isclose(std, 0.1, atol=0.1)
            # atol - абсолютный допуск

std_maks = (np.isclose(std, 0.1, rtol=0.01)
            # rtol - долевой допуск

frame = frame.rename(columns={"старое_имя": "новое_имя"})

status = frame["health_status"]
status.ne("dead")
# != для цепочек

frame["died"] = status.str.startswith(DIED_PREFIX, na=False)
# превращает na в False чтобы na не ломали сумирование

inside = participants[participants["in_vehicle"]]
frame = inside.merge(
    vehicles[["source_id", "vehicle_idx", "brand", "year", "category"]],
    on=["source_id", "vehicle_idx"],
    how="inner",
    validate="many_to_one",
)
# sql merge — по значениям колонок.

frame = pd.concat([counts.loc[~rare], rest], ignore_index=True)
# склеить
# ignore_index - выбрасывает старые и делают новые 0 -> ...

X = np.hstack([X_num, X_cat])
# склеивает numpy-массивы по горизонтали (добавляет столбцы), np.vstack - по вертикали (добавляет строки)
# индексов и имён столбцов нет, строки склеиваются просто по позиции
# число строк должно совпадать, иначе ValueError
# аналог в pandas: pd.concat([a, b], axis=1), но он выравнивает по индексу

frame = df[[f"pc{i}" for i in range(1, pc_count+1)]]

len(df)          # строки
len(df.columns)  # колонки
df.shape         # кортеж (строки, колонки), то есть (11, 11)
df.size          # всего клеток

frame["cumulative"] = variance.cumsum()
# трейловая сумма

frame = pd.DataFrame(coords[:, :pc_count], columns=[f"pc{i}" for i in range(1, pc_count+1)], index=h3_profiles.index)
# генератор списка где строки через f"pc{i}"

X = df.drop(columns="count")

profiles.insert(0, "count", counts)
# insert(позиция, имя, значения) - вставляет колонку на нужную позицию, остальные сдвигаются вправо
# работает in place, возвращает None (не писать profiles = profiles.insert(...))
# если колонка с таким именем уже есть - ValueError (allow_duplicates=True чтобы разрешить)

df = df.assign(resid=x)
# assign(имя=значения) - добавляет колонку в конец, имя задается именем аргумента
# то же, что df["resid"] = x, но возвращает новую копию, исходный df не меняется
# можно передать функцию: df.assign(resid=lambda d: d.y - d.pred) - удобно в цепочках методов

# по смылу
# axis=0 - результат снизу
# axis=1 - результат справа

# axis — это «вдоль какой оси работает операция». axis=0 — строки (вертикаль), axis=1 — колонки (горизонталь).
#
pd.concat([a, b], axis=0)  # таблицы ставятся друг под друга, строк становится больше. МОЖНО - axis="rows"
# Совмещаются по колонкам: одинаковые имена сойдутся, разные дадут NaN.
pd.concat([a, b], axis=1)  # таблицы ставятся бок о бок, колонок становится больше. МОЖНО - axis="columns"
# Совмещаются по индексу: строка с индексом 87118...a из первой таблицы встанет напротив такой же из второй.


h3_category = pd.crosstab(frame["cell"], frame["category"], normalize="index")
# category   Наезд на пешехода  Столкновение  Съезд с дороги
# cell
# 87118...a               0.40          0.55            0.05
# 87118...b               0.06          0.24            0.70
# normalize - процент по ирдексу (строчке)


def dtp_category_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = (
        df.groupby("category", dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False, ignore_index=True)
    )
    return frame

# сдесь ignore index == reset_index после сортировки

# ТОЖЕ САМОЕ только сначала индексы сортируем ибо value_count по убыванию count
frame = frame["year_month"].value_counts().sort_index().reset_index()

counts = (
        pd.Series(cells, name="cell")
        .value_counts()
        .rename("count")
        .reset_index()
    )

counts = frame["cell"].value_counts()
frame = frame[frame["cell"].isin(counts[counts >= min_count].index)]
# тоже самое что size() в groupby только для series
# схлопывает повторы - возвращает series где index то что схлопнули и столбец count
df.groupby("cell").size()        # Series: ячейка → сколько строк
df["cell"].value_counts()         # Series: ячейка → сколько строк

Series.between(left, right)
# возвращает булеву Series: True там, где значение попало в диапазон.
# По умолчанию границы включаются — то есть это (s >= left) & (s <= right)

pivot = frame.pivot(index="week_day", columns="hour", values="count")
# разворачивает как мы хотим

# pandas объектно-ориентирован — почти всё метод.
df.sum()  df.dropna()  s.isna()  s.astype()  s.str.strip()
# Функций модуля мало, и они трёх видов: создание + объединение + pd.isna()
pd.read_csv, pd.to_numeric, pd.to_datetime, pd.DataFrame

# numpy функционален — почти всё np.*. (функция модуля)
np.isfinite(a)  np.log(a)  np.where(m)  np.diff(a)  np.concatenate([a, b])
# Методов у ndarray мало, и они про сам массив, а не про математику:
a.sum()  a.mean()  a.max()  a.reshape()  a.astype()  a.copy()  a.T  a.argmax()

# --- Series.duplicated(keep='first') ---
# Возвращает Series типа BOOL той же длины и с тем же индексом.
# True = значение уже встречалось ранее (по правилу keep), т.е. это дубликат.
# Сравнение по ЗНАЧЕНИЮ, индекс не участвует. NaN сравнивается с NaN как равный.
s.duplicated()                  # keep='first': первое вхождение False, остальные True
s.duplicated(keep='last')       # последнее вхождение False, остальные True
s.duplicated(keep=False)        # True у ВСЕХ вхождений, у которых есть дубли
type(s.duplicated())            # <class 'pandas.core.series.Series'>, dtype=bool

# Число дубликатов = сумма True (bool суммируется как 0/1)
s.duplicated().sum()            # 2

# --- DataFrame.duplicated(subset=None, keep='first') ---
# Возвращает Series типа BOOL длиной = число строк, индекс = индекс DataFrame.
# ВАЖНО: возвращает НЕ DataFrame. Дубликат определяется по КОМБИНАЦИИ значений строки.
# subset=None -> учитываются ВСЕ столбцы.
df.duplicated()                 # строки 1 и 3 дублируют строку 0
type(df.duplicated())           # <class 'pandas.core.series.Series'>, dtype=bool

# subset = список/имя столбцов -> дубликаты ищутся только по ним
df.duplicated(subset=["a"])
df.duplicated(subset="a")       # можно строкой, если один столбец

# keep работает так же, как у Series
df.duplicated(keep='last')
df.duplicated(keep=False)

# --- Типовое применение: булева маска для фильтрации ---
df[df.duplicated()]             # только строки-дубликаты
df[~df.duplicated()]            # эквивалент df.drop_duplicates()

# --- Отличие от drop_duplicates ---
# duplicated()      -> Series[bool] (МАСКА, что удалять)
# drop_duplicates() -> DataFrame/Series (уже ОЧИЩЕННЫЕ данные), те же параметры subset/keep

# ИТОГ:
# duplicated -> ВСЕГДА Series[bool] той же длины (и для Series, и для DataFrame)
# keep='first' (по умолчанию) | 'last' | False
# subset — только у DataFrame

df.to_string
series.tolist
# pandas функции

# --- Series.unique() ---
# Возвращает МАССИВ уникальных значений в порядке первого появления (без сортировки).
# NaN СЧИТАЕТСЯ значением и попадает в результат ровно один раз. Параметров нет.

# --- Series.nunique(dropna=True) ---
# Возвращает int — количество различных значений.
# По умолчанию dropna=True: NaN НЕ учитывается. dropna=False добавляет +1 за NaN.

# --- DataFrame.unique() ---
# НЕ СУЩЕСТВУЕТ -> AttributeError

# --- DataFrame.nunique(axis=0, dropna=True) ---
# Возвращает Series из int.
# axis=0 (по умолчанию): индекс = имена столбцов, значения = число уникальных в столбце.
# axis=1: индекс = индекс строк, значения = число уникальных в строке.

# ============================================================
#  1. ИЕРАРХИЯ ТИПОВ
# ============================================================

# DataFrame = набор Series {} словарь где ключи - атрибуты, значения [], (названия колонок), значения Series которые равняются по внутреннему индексу
# Series = ndarray(одномерный) [] + индекс (метки строк) (можно создать из словаря тогда ключи - атрибуты
# ndarray = плотный блок памяти одного типа (пихнуть можно что угодно но приведет к одному) (n dimensional)

df = pd.read_csv("data.csv")     # pandas.DataFrame
df["price"]                      # pandas.Series  — ОДИН столбец внутри всегда одномерный ndarray
df[["price", "area"]]            # DataFrame — двумерный (двойные скобки) таблица
df["price"].to_numpy()           # numpy.ndarray — массив любой dimensional

df.median(numeric_only=True)

# np.array — функция (конструктор) copy=True по умолчанию
# np.ndarray — тип того, что она создаёт

a = np.array([1., 2., 3.])
len(a)      # 3
a.size      # 3
a.shape     # (3,)

s.size       # 5 — всего элементов, включая NaN
s.count()    # 3 — только непустые значения

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
df.to_string() # все показать
df.shape       # (строк, столбцов) - tuple
df.columns     # название колонок - возвращает Index - хитрый массив с хеш таблицей
df.columns[mask]

df.columns      # тип КАЖДОЙ колонки
# dtype определяется СОДЕРЖИМЫМ:
# одна ячейка "n/a" → вся колонка станет object (строки) → арифметика не работает
# int64 float64 bool object

df.info()      # типы + сколько непустых - void
df.describe()  # статистика по числовым - df

dupl = df.duplicated(keep="first").sum()
# duplicated() - Series T/F
df.sum()          # Series по колонкам
df.sum().sum()    # одно число


# axis=k — это ось, которая исчезает. Работает для любой размерности:
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

df.loc[5, "price"]     # по МЕТКАМ (строка, столбец) индекс, фича
passed = df.loc[df['scores'] >= 60, 'student'] # series
# можно прокинуть и булеву маску на нужные индексы

mask = df["scores"] >= 60
student = df["student"]
passed = student[mask]

threshold = counts.loc[counts["category"] == "Иной вид ДТП", "count"].iloc[0]
# забрать конкретное значение ячейки

df.iloc[5, 0]          # по ПОЗИЦИЯМ (строка, столбец)
# базовая индексация int 0 -> n-1 может быть изменена но iloc всегда идет по дефолтной 0 1 ...
df.iloc[124:]  # с 124 строки и до конца

# Индекс = метки строк. После фильтрации становится дырявым: 0,3,7,12...
filtered = df[df["price"] > 0].reset_index(drop=True)
df_copy = df_copy.sort_values(by=["date", "market"]).reset_index(drop=True)
# Список нужен только когда ключей несколько:
# если колонок несколько то списком пихаем а не просто через запятую

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
# nan - numpy в pandas напечатет NaN принято nan
# NA и NaT - pandasi
# pd.NA - не привязан к float заменяем на него когда string

np.nan   # float-колонки, numpy-массивы — классика
pd.NA    # nullable-типы: Int64, string, boolean
pd.NaT   # datetime
None     # object-колонки, Python-код

.str  # строки
df["point_longitude"] = df["point"].str.get("long")  # применяем str чтобы обртится к dict
.cat  # категориональные данные
.dt   # дата время
df_copy["year"] = df_copy["date"].dt.year
df_copy["month"] = df_copy["date"].dt.month

frame["year_month"] = datetime.dt.to_period("M")
# получить хороший формат yyyy-mm который сраниватся и тд
# Это называется "accessor" (аксессор/накопитель методов).a
# Вне pd np не существует
# необходимо юзать с Series, чтобы выполнить операцию нужную
(date - df_copy["date"])          # Timedelta
(date - df_copy["date"]).dt.days  # int64

blank = market.str.strip() == ""
non_valid = market.isna() | blank
#.str.strip() убирает пробелы по краям возвращает series, дальше скалярное сравнение с пустой строкой.
astype(str) # нужен, чтобы .str не спотыкался, если в колонке смешаны типы.

parsed = pd.to_datetime(date, format="%Y-%m-%d", errors="coerce")
parsed = pd.to_datetime(raw, errors="coerce", format="mixed") # пытется распарсить любые виды дт
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
# groupby - ленивый ничего не делает без терминальной функции (сортирует по умолчанию, выбрасывание NaN)
# возвращает Series чтобы вернуться к DF надо reset_index() и потом только сортировать
# Читается: сгруппируй по market → возьми price → посчитай среднее в каждой группе.
# market
# Herat 75.0
# Kabul 110.0

s = df.groupby(keys).agg(
    имя_результата=("исходная_колонка", "функция"),
    ...
)
# на выходе схлопнутый (сгруппированный) series где сначала колонки из keys потом колонки агрегации
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
x in s             # ищет по ИНДЕКСУ, не по значениям    s.isin([...])
if s:              # ValueError: ambiguous               s.any() / s.all() / s.empty
and / or / not     # ValueError: ambiguous               &  |  ~   (+ скобки!)
len(df)            # работает — число строк              df.shape[0] явнее
max(s), sum(s)     # работают, но Python-цикл            s.max(), s.sum()
sorted(s)          # вернёт list, потеряет индекс        s.sort_values()
str(s)             # строка всего объекта                s.astype(str)
round(s, 2)        # работает через __round__            s.round(2)
abs(s)             # работает                            s.abs()
reversed(s)        # не то                               s[::-1]

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


lat = lat.where(lat.notna(), pd.to_numeric(coord_frame["point_latitude"], errors="coerce"))
# Итог: «возьми geometry, а если её нет — возьми point». Обычный fallback, векторизованный.

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
