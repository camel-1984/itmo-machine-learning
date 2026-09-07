from src.python import report

# import src.syntax работает, но
# заставляет писать полный путь при каждом вызове.
# src.syntax.report(...)

# import numpy as np - импорт под псевдоним

def main():
    predictions = [0.9, 0.2, 0.7, 0.4, 0.85]
    labels = [1, 0, 1, 1, 0]
    report(predictions, labels)

if __name__ == "__main__":
    main()