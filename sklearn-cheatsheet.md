# scikit-learn: шпаргалка

Библиотека классического ML: предобработка, модели, метрики, валидация. Нейросети — не сюда (это PyTorch и т.п.). Импортируется как `sklearn`, ставится как `scikit-learn`.

Вся библиотека держится на **одном интерфейсе**: любой объект — это класс с методами `fit` / `transform` / `predict`. Выучил его один раз — умеешь пользоваться всеми сотнями классов.

---

## 1. Данные на входе

- `X` — признаки, 2D: `(n_samples, n_features)`. DataFrame или ndarray.
- `y` — целевая переменная, 1D: `(n_samples,)`. Series или ndarray.
- Строка = объект (гексагон, клиент, ДТП), колонка = признак.

```python
X = df.drop(columns="target")
y = df["target"]
```

Одна колонка как `X` — всё равно 2D:

```python
X = df[["area"]]     # DataFrame, (n, 1) — правильно
X = df["area"]       # Series, (n,) — ValueError: Expected 2D array
```

sklearn **не умеет** в строки и NaN (почти все классы падают). Их надо закодировать и заполнить до модели — см. раздел 4.

---

## 2. Интерфейс Estimator

Estimator — любой объект sklearn, который учится на данных.

```python
obj = StandardScaler()      # 1. создать, гиперпараметры — в конструктор
obj.fit(X_train)            # 2. обучить: запомнить что-то о данных
obj.transform(X_test)       # 3a. трансформер: вернуть преобразованные X
model.predict(X_test)       # 3b. модель: вернуть предсказания y
```

| Метод | У кого | Что делает |
|---|---|---|
| `fit(X, y)` | у всех | учится; возвращает **сам объект** (можно цепочкой) |
| `transform(X)` | трансформеры (scaler, encoder, PCA) | возвращает новый X |
| `fit_transform(X)` | трансформеры | `fit` + `transform` за один вызов |
| `predict(X)` | модели | возвращает предсказанные y |
| `predict_proba(X)` | классификаторы | вероятности классов, `(n, n_classes)` |
| `score(X, y)` | модели | метрика по умолчанию (accuracy / R²) |

### Атрибуты с `_` на конце

Всё, что объект **выучил** в `fit`, лежит в атрибутах с подчёркиванием в конце:

```python
scaler.mean_, scaler.scale_
pca.explained_variance_ratio_, pca.components_
model.coef_, model.intercept_
model.feature_importances_
```

Нет `_` — это гиперпараметр, заданный руками. Есть `_` — результат обучения. До `fit` таких атрибутов нет → `NotFittedError` / `AttributeError`.

### Что возвращается

`transform` / `predict` возвращают **ndarray**, имена колонок и индекс теряются. Вернуть обратно:

```python
pd.DataFrame(x_scaled, columns=X.columns, index=X.index)
```

Или один раз на весь проект:

```python
from sklearn import set_config
set_config(transform_output="pandas")   # трансформеры отдают DataFrame
```

---

## 3. Train / test и утечка данных

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)
```

- `random_state` — фиксирует случайность, результат воспроизводится. Ставь всегда.
- `stratify=y` — доли классов в train и test одинаковые. Для классификации — почти всегда.
- Временные ряды **не перемешивают**: train — прошлое, test — будущее (`shuffle=False` или `TimeSeriesSplit`).

### Главное правило

**`fit` — только на train. На test — только `transform` / `predict`.**

```python
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)   # учим среднее и std на train
X_test_s = scaler.transform(X_test)          # применяем ТЕ ЖЕ среднее и std
```

Ошибка: `fit_transform` на всём `X` до разбиения или отдельно на test. Модель «подсматривает» в test → метрика завышена, в проде будет хуже. Это называется **data leakage** (утечка). Лечится `Pipeline` (раздел 5) — там перепутать невозможно.

Исключение: исследование без модели (PCA для картинки, кластеры для EDA) — там test нет, `fit_transform` на всём нормально. Как в `src/pca.py`.

---

## 4. Предобработка

Всё в `sklearn.preprocessing` и `sklearn.impute`.

### Масштабирование

| Класс | Что делает | Когда |
|---|---|---|
| `StandardScaler` | `(x - mean) / std` → среднее 0, std 1 | по умолчанию; обязательно для PCA, KNN, линейных, SVM |
| `MinMaxScaler` | в диапазон `[0, 1]` | нужны ограниченные значения |
| `RobustScaler` | через медиану и IQR | много выбросов |

Деревьям (DecisionTree, RandomForest, бустинги) масштаб **не нужен** — они сравнивают значения с порогом, а не считают расстояния.

Почему PCA без scaler врёт: признак в рублях (разброс 10⁶) задавит признак в штуках (разброс 10) — компонента просто повторит «самую большую» колонку.

### Категории → числа

```python
OneHotEncoder(handle_unknown="ignore")   # цвет → 3 колонки 0/1; без порядка
OrdinalEncoder()                          # низкий/средний/высокий → 0/1/2; есть порядок
```

`handle_unknown="ignore"` — категория, которой не было в train, не роняет `transform` на test.

`LabelEncoder` — только для `y`, не для признаков.

### Пропуски

```python
SimpleImputer(strategy="median")          # числа
SimpleImputer(strategy="most_frequent")   # категории
```

Среднее / медиана считаются на train (это тоже `fit`) — правило утечки действует.

---

## 5. Pipeline и ColumnTransformer

### Pipeline — цепочка шагов

```python
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe.fit(X_train, y_train)     # scaler.fit_transform → model.fit
pipe.predict(X_test)           # scaler.transform → model.predict
```

Весь pipeline ведёт себя как **одна модель**: у него те же `fit` / `predict`. Утечка невозможна — pipeline сам вызывает правильный метод на каждом шаге.

### ColumnTransformer — разным колонкам разная обработка

```python
from sklearn.compose import make_column_transformer

pre = make_column_transformer(
    (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),
    (OneHotEncoder(handle_unknown="ignore"), cat_cols),
)
pipe = make_pipeline(pre, RandomForestClassifier(random_state=0))
```

Это стандартная форма почти любого табличного проекта: `ColumnTransformer` → модель, всё в одном `Pipeline`.

`make_pipeline` / `make_column_transformer` сами придумывают имена шагов. Полные `Pipeline([("name", obj), ...])` нужны, когда к шагу надо обращаться по имени (например, в `GridSearchCV`).

---

## 6. Модели

| Задача | Класс | Заметки |
|---|---|---|
| Регрессия | `LinearRegression` | базовая линия, `coef_` читаются |
| Регрессия | `Ridge`, `Lasso` | линейная + регуляризация `alpha` |
| Классификация | `LogisticRegression` | несмотря на имя — классификатор; первое, что пробуют |
| Оба | `KNeighborsClassifier` / `Regressor` | по соседям; нужен scaler |
| Оба | `DecisionTreeClassifier` / `Regressor` | интерпретируемо, легко переобучается |
| Оба | `RandomForestClassifier` / `Regressor` | сильный baseline на табличках без настройки |
| Оба | `GradientBoostingClassifier`, `HistGradientBoosting...` | обычно лучший результат на табличках |
| Кластеризация | `KMeans` | нет `y`; `fit(X)`, метки в `labels_` |
| Снижение размерности | `PCA` | трансформер, см. `src/pca.py` |

Импорты разбросаны по модулям: `sklearn.linear_model`, `sklearn.neighbors`, `sklearn.tree`, `sklearn.ensemble`, `sklearn.cluster`, `sklearn.decomposition`.

**Baseline первым.** Прежде чем крутить модель, посчитай, сколько даёт тупое решение: `DummyClassifier(strategy="most_frequent")` / `DummyRegressor()`. Accuracy 95% ничего не значит, если 95% объектов — один класс.

---

## 7. Метрики

`sklearn.metrics`, сигнатура всегда `metric(y_true, y_pred)` — сначала правда, потом предсказание.

### Классификация

| Метрика | Смысл |
|---|---|
| `accuracy_score` | доля верных; врёт при дисбалансе классов |
| `precision_score` | из предсказанных «1» — сколько реально 1 |
| `recall_score` | из реальных 1 — сколько нашли |
| `f1_score` | среднее гармоническое precision и recall |
| `roc_auc_score` | качество ранжирования; на вход — `predict_proba(X)[:, 1]`, не `predict` |
| `confusion_matrix` | таблица ошибок: откуда что берётся |
| `classification_report` | всё сразу текстом по каждому классу |

### Регрессия

| Метрика | Смысл |
|---|---|
| `mean_absolute_error` | средняя ошибка в единицах `y`; легко объяснить |
| `mean_squared_error` | сильнее штрафует большие ошибки; `root_mean_squared_error` — в единицах `y` |
| `r2_score` | доля объяснённой дисперсии; 1 — идеал, 0 — как среднее, < 0 — хуже среднего |

---

## 8. Кросс-валидация и подбор гиперпараметров

Одно разбиение train/test — шумная оценка: повезло/не повезло с разбиением. Кросс-валидация делит train на k частей и обучает k раз.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1")
scores.mean(), scores.std()
```

Передаётся **весь pipeline**, а не голая модель — иначе scaler учится на всех фолдах сразу (утечка).

### GridSearchCV

```python
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(
    pipe,
    param_grid={"randomforestclassifier__n_estimators": [100, 300],
                "randomforestclassifier__max_depth": [None, 10]},
    cv=5, scoring="f1",
)
grid.fit(X_train, y_train)
grid.best_params_, grid.best_score_
grid.best_estimator_        # уже переобучен на всём train с лучшими параметрами
```

- Имя параметра внутри pipeline: `имяшага__параметр` (два подчёркивания).
- Перебор большой → `RandomizedSearchCV` (пробует случайные комбинации).

### Порядок

1. `train_test_split` — test откладывается и **не трогается**.
2. Подбор параметров — кросс-валидацией **только на train**.
3. Финальная оценка — один раз на test.

Если смотрел на test и после этого что-то менял — test перестал быть честной оценкой.

---

## 9. Типичные грабли

- `fit` на test или на всём датасете до разбиения — утечка.
- Scaler и модель отдельно, а в `cross_val_score` только модель — утечка.
- Забыл `random_state` — результаты скачут от запуска к запуску.
- Accuracy на несбалансированных классах — смотри precision / recall / F1 и `confusion_matrix`.
- `roc_auc_score(y, model.predict(X))` — нужны вероятности, а не метки.
- Признак, который известен только после события (например, «число погибших» при предсказании тяжести ДТП), — утечка через сами данные, никакой pipeline не спасёт.
- `X` одномерный — `df["col"]` вместо `df[["col"]]`.
