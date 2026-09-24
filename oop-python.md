# ООП в Python (для тех, кто пришёл из Java)

Главная идея, которую надо принять сразу: **Python доверяет программисту**.
Java защищает тебя компилятором (типы, `private`, `final`, обязательные объявления полей).
Python почти ничего не запрещает — вместо запретов есть **соглашения**, которые все соблюдают.
Девиз сообщества: *"We are all consenting adults here"* — «мы все тут взрослые люди».

---

## 1. Самый простой класс

```python
class Dog:
    pass

d = Dog()        # нет слова new
print(type(d))   # <class '__main__.Dog'>
```

- `new` не нужен — класс вызывается как функция.
- `pass` — «пустое тело» (в Python нельзя оставить блок пустым).
- Все классы неявно наследуются от `object` (как в Java).

---

## 2. Конструктор и `self`

```python
class Dog:
    def __init__(self, name, age):
        self.name = name     # создаём поле экземпляра
        self.age = age

    def bark(self):
        print(f"{self.name}: Гав!")

d = Dog("Рекс", 3)
d.bark()
```

| Java | Python |
|---|---|
| `this` (неявный) | `self` (явный **первый параметр** каждого метода) |
| `this.name = name` | `self.name = name` |
| конструктор = имя класса | `__init__` |

**Про `self`:**
- Это просто имя первого параметра, а не ключевое слово (можно назвать `this`, но так никто не делает — не надо).
- `d.bark()` — это на самом деле `Dog.bark(d)`. Python сам подставляет объект первым аргументом.
- Внутри метода к полям **всегда** обращаемся через `self.` — просто `name` будет искаться как локальная/глобальная переменная.

**Про `__init__`:**
- Строго говоря, это не конструктор, а **инициализатор**: объект уже создан (это делает `__new__`), `__init__` только заполняет его. На практике `__new__` почти никогда не трогают.
- Конструктор может быть только **один** (перегрузки нет, см. раздел 6).

---

## 3. Почему поля объявляются в конструкторе (и вообще где угодно)

### Как это устроено

В Java поля — часть **описания класса**: компилятор заранее знает структуру объекта.

В Python объект — это по сути **словарь** атрибутов:

```python
d = Dog("Рекс", 3)
print(d.__dict__)   # {'name': 'Рекс', 'age': 3}
```

`self.name = name` — это не «объявление», а просто **запись в словарь** `d.__dict__['name'] = name`.
Поэтому:

```python
d.color = "black"       # новое поле извне — работает!
print(d.__dict__)       # {'name': 'Рекс', 'age': 3, 'color': 'black'}

d2 = Dog("Бобик", 5)
d2.color                # AttributeError — у d2 такого поля нет

del d.age               # поле можно даже удалить
```

У каждого объекта **свой** набор полей. Поле существует, только когда в него что-то присвоили.

### Как принято на практике

- **Все поля создавай в `__init__`**, даже если значения пока нет — ставь `None`:
  ```python
  def __init__(self, name):
      self.name = name
      self.owner = None   # появится позже, но «объявлено» тут
  ```
  Тогда, открыв `__init__`, читатель видит всю структуру объекта. PyCharm подсвечивает поля, созданные вне `__init__`.
- Не добавляй поля извне (`d.color = ...`) в обычном коде — это возможность языка, а не рекомендуемая практика.
- Для документирования структуры используют **аннотации типов** (раздел 11) и **dataclass** (раздел 10).

### Как запретить добавлять новые поля: `__slots__`

```python
class Point:
    __slots__ = ("x", "y")

    def __init__(self, x, y):
        self.x = x
        self.y = y

p = Point(1, 2)
p.z = 3   # AttributeError: 'Point' object has no attribute 'z'
```

- Объект больше не имеет `__dict__`, поля хранятся компактнее → меньше памяти, чуть быстрее доступ.
- Используют в основном ради **памяти** (миллионы мелких объектов), а не ради «строгости».

---

## 4. Атрибуты класса vs атрибуты экземпляра (аналог `static`)

```python
class Dog:
    species = "Canis familiaris"   # атрибут КЛАССА (общий, как static)
    count = 0

    def __init__(self, name):
        self.name = name           # атрибут ЭКЗЕМПЛЯРА
        Dog.count += 1

a = Dog("A")
b = Dog("B")
print(Dog.count)     # 2
print(a.species)     # "Canis familiaris" — читается через экземпляр тоже
```

### Как ищется атрибут

`a.species` → сначала ищем в `a.__dict__` → не нашли → ищем в классе `Dog` → потом в родителях.

### Грабли №1: присваивание через экземпляр

```python
a.species = "wolf"   # НЕ меняет атрибут класса!
                     # создаёт НОВОЕ поле в a.__dict__, которое «затеняет» классовое
print(a.species)     # wolf
print(b.species)     # Canis familiaris
print(Dog.species)   # Canis familiaris
```

Менять общий атрибут — только через класс: `Dog.species = ...`.

### Грабли №2: изменяемый атрибут класса

```python
class Dog:
    tricks = []                 # ОДИН список на все объекты!

    def add_trick(self, t):
        self.tricks.append(t)   # не присваивание, а изменение → меняем общий список

a, b = Dog(), Dog()
a.add_trick("сидеть")
print(b.tricks)   # ['сидеть'] — сюрприз
```

**Правило:** списки, словари и прочие изменяемые значения создавай в `__init__` (`self.tricks = []`).

> Родственные грабли — изменяемый аргумент по умолчанию: `def __init__(self, items=[])` —
> этот список создаётся **один раз** при определении функции и общий для всех вызовов.
> Правильно: `items=None`, а внутри `self.items = items if items is not None else []`.

---

## 5. Модификаторы доступа: их нет

В Python **нет** `private`, `protected`, `public`. Всё публично. Вместо этого — соглашения по именам.

### `name` — публичное

Обычное поле, пользуйся.

### `_name` — «внутреннее, не трогай» (аналог private/protected по смыслу)

```python
class Account:
    def __init__(self, balance):
        self._balance = balance

acc = Account(100)
acc._balance   # 100 — работает, никто не запретит
```

- Одно подчёркивание — сигнал другим программистам: «это деталь реализации, может измениться, не используй снаружи».
- Язык не мешает, но IDE и линтеры предупредят. В реальном коде это **основной** способ сделать «приватное».
- Бонус: `from module import *` не импортирует имена с `_`.

### `__name` — name mangling (искажение имени)

```python
class Account:
    def __init__(self, balance):
        self.__balance = balance

acc = Account(100)
acc.__balance            # AttributeError
acc._Account__balance    # 100 — вот оно
print(acc.__dict__)      # {'_Account__balance': 100}
```

- Python переименовывает `__balance` в `_Account__balance` (добавляет имя класса).
- **Цель — не защита**, а избежание конфликтов имён при наследовании: если в подклассе тоже будет `__balance`, он станет `_SubClass__balance` и не перезатрёт родительский.
- На практике используется редко. По умолчанию пиши `_name`.

### `__name__` — dunder (double underscore) методы

Имена с двумя подчёркиваниями **с обеих сторон** — специальные методы языка (`__init__`, `__str__`, `__len__`...). Это не «приватное», наоборот — их вызывает сам Python. Свои такие имена придумывать не надо.

| Java | Python |
|---|---|
| `public` | `name` |
| `protected` / `private` | `_name` (соглашение) |
| — | `__name` (защита от конфликтов при наследовании) |

---

## 6. Геттеры и сеттеры: `@property`

В Java пишут `getX()/setX()` заранее «на всякий случай», потому что потом поменять публичное поле на метод — сломать всех пользователей.

В Python так **не делают**. Начинай с обычного публичного поля. Если понадобилась логика — превращаешь его в `property`, и **код снаружи не меняется**:

```python
class Temperature:
    def __init__(self, celsius):
        self.celsius = celsius        # тут уже вызывается сеттер!

    @property
    def celsius(self):                # геттер
        return self._celsius

    @celsius.setter
    def celsius(self, value):         # сеттер
        if value < -273.15:
            raise ValueError("Ниже абсолютного нуля")
        self._celsius = value

    @property
    def fahrenheit(self):             # вычисляемое поле, только чтение
        return self._celsius * 9 / 5 + 32

t = Temperature(20)
t.celsius = 25        # выглядит как поле, но работает сеттер
print(t.fahrenheit)   # 77.0
t.fahrenheit = 100    # AttributeError: нет сеттера → read-only
```

- Реальное значение хранится в `_celsius`, а `celsius` — «фасад».
- Так делается read-only поле: только `@property` без сеттера.
- Есть ещё `@x.deleter` — используется редко.

---

## 7. Виды методов

```python
class Date:
    def __init__(self, day, month, year):
        self.day, self.month, self.year = day, month, year

    # 1. Обычный метод — получает экземпляр (self)
    def show(self):
        return f"{self.day:02}.{self.month:02}.{self.year}"

    # 2. Метод класса — получает сам класс (cls)
    @classmethod
    def from_string(cls, s):
        d, m, y = map(int, s.split("."))
        return cls(d, m, y)          # cls, а не Date — корректно для наследников

    # 3. Статический метод — ничего не получает, просто функция внутри класса
    @staticmethod
    def is_valid(day, month):
        return 1 <= day <= 31 and 1 <= month <= 12

d = Date.from_string("24.09.2026")
Date.is_valid(31, 12)
```

| Вид | Первый параметр | Когда использовать |
|---|---|---|
| обычный | `self` | работает с данными объекта |
| `@classmethod` | `cls` | **альтернативные конструкторы**, работа с атрибутами класса |
| `@staticmethod` | — | вспомогательная функция, логически связанная с классом |

### Перегрузки методов нет

Если определить два метода с одним именем — второй просто перезапишет первый. Вместо перегрузки:

- **аргументы по умолчанию**: `def __init__(self, x=0, y=0)`
- **именованные аргументы**: `Point(x=1, y=2)`
- `*args, **kwargs` — произвольное число аргументов
- **`@classmethod`-конструкторы** с понятными именами: `Date.from_string(...)`, `Date.today()` — это идиоматичная замена перегрузке конструкторов.

> Кстати, в Python часто вообще не делают статических методов: если функции не нужен класс,
> её просто пишут на уровне модуля. В Python не обязательно всё засовывать в класс.

---

## 8. Наследование

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return "..."

class Dog(Animal):                    # extends → скобки
    def __init__(self, name, breed):
        super().__init__(name)        # вызов конструктора родителя
        self.breed = breed

    def speak(self):                  # переопределение (никакого @Override)
        return "Гав"
```

- `super().__init__(...)` **не вызывается автоматически**. Забыл — поля родителя не создадутся (в Java компилятор вставляет `super()` сам).
- Все методы «виртуальные» — переопределяются всегда, `final` нет (есть `typing.final`, но это только подсказка для анализаторов).
- Проверки: `isinstance(d, Animal)` → `True`, `issubclass(Dog, Animal)` → `True`.

### Множественное наследование

В Java — только от одного класса. В Python — сколько угодно:

```python
class Flyer:
    def fly(self): return "лечу"

class Swimmer:
    def swim(self): return "плыву"

class Duck(Animal, Flyer, Swimmer):
    pass
```

**MRO** (Method Resolution Order — порядок поиска методов): в каком порядке Python ищет метод в родителях.

```python
print(Duck.__mro__)
# (Duck, Animal, Flyer, Swimmer, object)
```

- `super()` идёт **не к «родителю»**, а к **следующему классу в MRO**. Это важно при множественном наследовании.
- На практике используют в основном **миксины** — маленькие классы, добавляющие одну способность (`JsonMixin`, `LoggingMixin`). Сложные «ромбы» наследования — признак плохого дизайна.

---

## 9. Интерфейсы и абстрактные классы

### Утиная типизация (duck typing)

*«Если оно крякает как утка — это утка».* Python не проверяет тип, он просто вызывает метод. Есть метод — работает.

```python
class Duck:
    def speak(self): return "Кря"

class Robot:
    def speak(self): return "Бип"

def make_noise(x):      # никакого интерфейса Speakable
    print(x.speak())

make_noise(Duck())
make_noise(Robot())
```

Интерфейсы в Java-смысле для этого **не нужны**. Это главное отличие мышления.

### Абстрактный класс: модуль `abc`

Когда нужно **гарантировать**, что наследник реализовал методы:

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        ...

    def describe(self):                   # обычный метод тоже можно
        return f"Площадь: {self.area()}"

class Square(Shape):
    def __init__(self, a):
        self.a = a
    def area(self):
        return self.a ** 2

Shape()      # TypeError: Can't instantiate abstract class
Square(3)    # ок
```

Если наследник не реализовал все `@abstractmethod` — его тоже нельзя создать. Проверка происходит при **создании объекта**, а не при компиляции.

### `Protocol` — интерфейс без наследования

```python
from typing import Protocol

class Speaker(Protocol):
    def speak(self) -> str: ...

def make_noise(x: Speaker) -> None:
    print(x.speak())
```

`Duck` и `Robot` **не наследуются** от `Speaker`, но статический анализатор (mypy, PyCharm) проверит, что у них есть `speak`. Это «утиная типизация с проверкой». Во время выполнения ничего не проверяется.

---

## 10. Магические (dunder) методы

Это способ встроиться в синтаксис языка: операторы, `print`, `len`, `for`, `with`, `[]`...

### Строковое представление

```python
class Point:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __repr__(self):             # для разработчика: отладка, консоль, списки
        return f"Point(x={self.x}, y={self.y})"

    def __str__(self):              # для пользователя: print(), str()
        return f"({self.x}, {self.y})"
```

- `__str__` ≈ `toString()`. Если `__str__` нет — используется `__repr__`.
- Минимум всегда пиши `__repr__`. Хорошая практика: чтобы по нему можно было воссоздать объект.

### Сравнение и хеш

```python
    def __eq__(self, other):        # ==  (аналог equals)
        if not isinstance(other, Point):
            return NotImplemented
        return self.x == other.x and self.y == other.y

    def __hash__(self):             # hashCode
        return hash((self.x, self.y))
```

- По умолчанию `==` сравнивает **идентичность** (как `==` для объектов в Java).
- `is` — проверка, что это **тот же объект** (как `==` в Java). `==` — сравнение значений (как `.equals()`).
- Если определил `__eq__`, но не `__hash__` — объект становится **нехешируемым** (нельзя в `set` и ключом `dict`). Python делает это специально, чтобы не было рассинхрона.
- `<, <=, >, >=` → `__lt__, __le__, __gt__, __ge__`. Можно определить `__eq__` и `__lt__`, а остальное сгенерирует декоратор `@functools.total_ordering`.

### Арифметика (перегрузка операторов — в Java её нет)

```python
    def __add__(self, other):       # p1 + p2
        return Point(self.x + other.x, self.y + other.y)

    def __mul__(self, k):           # p * 3
        return Point(self.x * k, self.y * k)

    def __rmul__(self, k):          # 3 * p  (r = reflected, когда объект справа)
        return self * k
```

Так работают numpy-массивы и pandas: `a + b` — это вызов `__add__`.

### Контейнеры

```python
class Deck:
    def __init__(self):
        self.cards = list(range(52))

    def __len__(self):              # len(deck)
        return len(self.cards)

    def __getitem__(self, i):       # deck[0], deck[1:5], и for card in deck
        return self.cards[i]

    def __contains__(self, card):   # 5 in deck
        return card in self.cards
```

Ещё: `__setitem__` (`d[i] = x`), `__delitem__`, `__iter__` / `__next__` (итераторы).

### Прочие полезные

| Метод | Что делает |
|---|---|
| `__call__` | объект можно вызвать как функцию: `obj(x)` |
| `__enter__`, `__exit__` | работа с `with` (аналог try-with-resources) |
| `__bool__` | как объект ведёт себя в `if obj:` |
| `__getattr__` | вызывается, если атрибут не найден обычным способом |

Пример `with`:

```python
class Timer:
    def __enter__(self):
        import time
        self.start = time.time()
        return self
    def __exit__(self, exc_type, exc, tb):
        import time
        print(f"{time.time() - self.start:.3f} c")

with Timer():
    sum(range(10**6))
```

---

## 11. Аннотации типов

Типы можно указывать, но **Python их не проверяет при выполнении** — это подсказки для IDE, mypy и читателя.

```python
class User:
    name: str                       # объявление поля с типом (значения нет!)
    age: int

    def __init__(self, name: str, age: int) -> None:
        self.name = name
        self.age = age

    def greet(self, other: "User") -> str:    # кавычки: класс ещё не определён
        return f"Привет, {other.name}"

User(123, "abc")   # выполнится без ошибок; ругнётся только PyCharm/mypy
```

- `name: str` на уровне класса без значения **не создаёт** атрибут — только аннотацию. Это ближе всего к «объявлению поля» как в Java.
- С Python 3.11 есть `Self` (`from typing import Self`) для методов, возвращающих свой класс.
- В реальных проектах аннотации пишут почти всегда — это стандарт.

---

## 12. `dataclass` — классы-данные без бойлерплейта

Аналог Java `record` / Lombok `@Data`. Используется **очень** часто.

```python
from dataclasses import dataclass, field

@dataclass
class Point:
    x: float
    y: float = 0.0
    tags: list[str] = field(default_factory=list)   # изменяемое значение по умолчанию — только так
```

Автоматически генерирует `__init__`, `__repr__`, `__eq__`.

```python
p = Point(1, 2)
print(p)              # Point(x=1, y=2, tags=[])
p == Point(1, 2)      # True
```

Полезные параметры:
- `@dataclass(frozen=True)` — неизменяемый объект (как `record`/`final`-поля), и он становится хешируемым.
- `@dataclass(order=True)` — добавляет `<, >` (сравнение по полям по порядку).
- `@dataclass(slots=True)` (3.10+) — добавляет `__slots__`.
- `__post_init__(self)` — метод для доп. проверок после автоматического `__init__`.

**Практика:** если класс в основном хранит данные — бери `dataclass`. Для валидации данных извне (JSON, API) часто используют библиотеку **pydantic**.

---

## 13. Enum

```python
from enum import Enum, auto

class Color(Enum):
    RED = auto()
    GREEN = auto()

Color.RED           # <Color.RED: 1>
Color.RED.name      # 'RED'
Color.RED.value     # 1
Color["GREEN"]      # доступ по имени
for c in Color: ... # перебор
```

---

## 14. Всё — объект (даже классы и функции)

```python
class Dog: pass

type(Dog)           # <class 'type'> — класс тоже объект, экземпляр класса type
x = Dog             # класс можно положить в переменную
obj = x()           # и создать через неё объект

def f(): pass
f.anything = 42     # у функции тоже можно создать атрибут
```

Полезные функции для работы с объектами (рефлексия — в Python она простая):

| Функция | Что делает |
|---|---|
| `type(obj)` | класс объекта |
| `isinstance(obj, Cls)` | аналог `instanceof` (учитывает наследование) |
| `hasattr(obj, "x")` | есть ли атрибут |
| `getattr(obj, "x", default)` | получить атрибут по строке |
| `setattr(obj, "x", 5)` | установить атрибут по строке |
| `vars(obj)` / `obj.__dict__` | все поля экземпляра |
| `dir(obj)` | все атрибуты, включая методы и унаследованные |

Метаклассы («классы классов», наследники `type`) существуют, но в обычной разработке почти не нужны — это инструмент для авторов фреймворков.

---

## 15. Композиция, инкапсуляция — как принято

- **Композиция вместо наследования** — то же правило, что и в Java. Глубокие иерархии в Python редки.
- **Инкапсуляция** держится на дисциплине: `_` для внутреннего + `property` для контролируемого доступа.
- **Не пиши классы без нужды.** Если у класса один метод и `__init__` — скорее всего, это должна быть функция. В Java «всё класс», в Python — нет.
- **EAFP** (*Easier to Ask Forgiveness than Permission*): вместо проверок `if hasattr(...)` чаще пишут `try/except AttributeError`. Питонический стиль — пробовать и ловить исключение.

---

## 16. Шпаргалка Java → Python

| Java | Python |
|---|---|
| `new Dog()` | `Dog()` |
| `this` | `self` (явный параметр) |
| конструктор `Dog(...)` | `__init__(self, ...)` |
| объявление полей в теле класса | присваивание `self.x = ...` в `__init__` (+ аннотации) |
| `static` поле | атрибут класса |
| `static` метод | `@staticmethod` / `@classmethod` |
| перегрузка конструкторов | аргументы по умолчанию, `@classmethod`-фабрики |
| `private` / `protected` | `_name` (соглашение), `__name` (mangling) |
| `getX()` / `setX()` | `@property` |
| `extends` | `class B(A):` |
| `implements` интерфейс | duck typing / `ABC` / `Protocol` |
| `abstract` | `ABC` + `@abstractmethod` |
| `super.method()` | `super().method()` |
| `@Override` | ничего (есть `typing.override` в 3.12, только для анализаторов) |
| `final` | нет (`typing.final` — подсказка) |
| `toString()` | `__str__` / `__repr__` |
| `equals()` / `hashCode()` | `__eq__` / `__hash__` |
| `a == b` (ссылки) | `a is b` |
| `a.equals(b)` | `a == b` |
| `instanceof` | `isinstance()` |
| `record` | `@dataclass(frozen=True)` |
| try-with-resources | `with` + `__enter__`/`__exit__` |
| `interface Comparable` | `__lt__` и др. |

---

## 17. Типичные грабли (коротко)

1. Забыл `self` в параметрах метода → `TypeError: takes 0 positional arguments but 1 was given`.
2. Написал `name` вместо `self.name` внутри метода → `NameError` или чужая переменная.
3. Изменяемый атрибут класса (`tricks = []`) — общий на всех.
4. Изменяемый аргумент по умолчанию (`def f(x=[])`).
5. `obj.class_attr = ...` создаёт поле экземпляра, а не меняет классовое.
6. Забыл `super().__init__()` в наследнике.
7. Определил `__eq__` без `__hash__` → объект нельзя положить в `set`/`dict`.
8. Опечатка `self.nmae = ...` не даст ошибку — просто создастся новое поле. Ловят линтеры, `__slots__`, dataclass.
9. `is` вместо `==` для сравнения значений (`is` используют только с `None`: `if x is None`).
