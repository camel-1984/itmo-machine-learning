# HW01. First Date with Data

Первая домашняя работа по NumPy и pandas на реальном датасете WFP Food Prices (Kenya).

**Дедлайн: 11.09.2026, 23:59 (МСК).**

Условие: [`HW01.md`](HW01.md)  
Реализация: [`solutions.py`](solutions.py)

## Repository Structure

```text
.
├── HW01.md
├── README.md
├── solutions.py
├── requirements.txt
├── data/
│   └── wfp_food_prices_ken.csv
├── tests/
└── .github/workflows/tests.yml
```

В `solutions.py` уже заданы сигнатуры функций. Не изменяйте их: замените `raise NotImplementedError` своей реализацией.

## Getting Started

Для работы требуется Python 3.11+.

Создать и активировать виртуальное окружение в Linux или macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

В Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Все зависимости рекомендуется устанавливать в виртуальное окружение, чтобы не изменять глобальное Python-окружение. Код и тесты также следует запускать из активированного виртуального окружения.

Установить зависимости:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Запустить все открытые тесты:

```bash
pytest -q
```

Запустить один файл тестов:

```bash
pytest -q tests/test_numpy.py
```

Запустить один конкретный тест:

```bash
pytest -q tests/test_numpy.py::test_prepare_price_series_basic
```

После завершения работы можно деактивировать окружение:

```bash
deactivate
```

## Before You Submit

Проверьте, что:

- `pytest -q` проходит локально;
- сигнатуры функций в `solutions.py` не изменены;
- в решении нет хардкода под конкретные рынки, товары или даты;
- в git не попали `.venv`, `__pycache__`, `.pytest_cache` и другие локальные файлы.

Помимо открытых тестов, решение может проверяться дополнительными тестами в CI.

## Useful Resources

- [NumPy quickstart](https://numpy.org/doc/stable/user/quickstart.html)
- [pandas DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)
- [pandas: indexing and selecting data](https://pandas.pydata.org/docs/user_guide/indexing.html)
- [pandas: group by](https://pandas.pydata.org/docs/user_guide/groupby.html)
- [pandas: time series / date functionality](https://pandas.pydata.org/docs/user_guide/timeseries.html)
- [pandas.DataFrame.pivot_table](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pivot_table.html)
- [Matplotlib quick start](https://matplotlib.org/stable/users/explain/quick_start.html)
- [pytest: getting started](https://docs.pytest.org/en/stable/getting-started.html)
- [Python `venv`](https://docs.python.org/3/library/venv.html)
