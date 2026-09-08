import pandas as pd

def f():
    s = pd.Series(["a", 2, 3], index=(1, 2, 3))
    s2 = pd.Series(["z", "v", 4.5], index=(1, 2 ,3))
    df = pd.DataFrame({"a": s, "b": s2})

    x = df.loc[1, "a"]
    y = df.iloc[1, 1]

    print(s)
    print(df)
    print(x)
    print(y)

def d():




if __name__ == "__main__":
    pass

