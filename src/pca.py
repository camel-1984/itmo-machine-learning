from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def pca(
        h3_profiles: pd.DataFrame,
        pc_count: int = 2
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:

    x = h3_profiles.drop(columns="count")
    x_scaled = StandardScaler().fit_transform(x)
    pca = PCA(random_state=0)

    coords = pca.fit_transform(x_scaled)
    variance = pca.explained_variance_ratio_

    frame = pd.DataFrame(coords[:, :pc_count],
                         columns=[f"pc{i}" for i in range(1, pc_count+1)],
                         index=h3_profiles.index)

    frame.insert(0, "count", h3_profiles["count"])

    weights = pd.DataFrame(
        pca.components_.T,
        index=x.columns,
        columns=[f"pc{i}" for i in range(1, len(pca.components_) + 1)],
    )

    return frame, variance, weights


# variance — доля дисперсии, которую объясняет каждая компонента (сумма = 1)
# array([3.40877969e-01, 1.68181612e-01, 1.42295132e-01, 1.08668200e-01,
#        8.87431126e-02, 5.52068520e-02, 4.39729173e-02, 3.35064925e-02,
#        1.77240765e-02, 8.23636328e-04, 4.56495503e-20])
# PC1 объясняет 34%, PC2 — 17%, вместе ~51% всей информации.
# Последнее значение ~0 => один признак линейно зависит от остальных.

# coords — все компоненты (сырой результат fit_transform, n_samples x n_features)
#                   PC1     PC2     PC3   ...
# 861188017ffffff  -2.31    0.84   -0.12
# 86118806fffffff   1.47   -1.20    0.55
# 8611880a7ffffff   0.09    2.03   -1.31

# frame — то же, но обрезано до pc_count и с возвращённым "count"
#                  count    pc1     pc2
# 861188017ffffff    143   -2.31    0.84
# 86118806fffffff     27    1.47   -1.20
# 8611880a7ffffff      5    0.09    2.03
# Строка = гексагон, число = его координата в новом пространстве.
# Знак и величина: чем дальше от 0, тем сильнее выражена компонента.

# weights (loadings) — вклад каждого исходного признака в компоненту
#                 pc1     pc2     pc3   ...
# cafe           0.42   -0.05    0.31
# school        -0.11    0.58   -0.02
# parking        0.38    0.12   -0.44
# Читается так: pc1 растёт там, где много cafe и parking => "коммерческая зона".
# Большой |вес| = признак определяет компоненту, около 0 = не влияет.
