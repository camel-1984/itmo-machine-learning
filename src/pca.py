from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def pca(
        h3_profiles: pd.DataFrame,
        pc_count: int = 2
) -> tuple[pd.DataFrame, np.ndarray]:

    x = h3_profiles.drop(columns="count")
    x_scaled = StandardScaler().fit_transform(x)
    pca = PCA(random_state=0)

    coords = pca.fit_transform(x_scaled)
    variance = pca.explained_variance_ratio_

    frame = pd.DataFrame(coords[:, :pc_count],
                         columns=[f"pc{i}" for i in range(1, pc_count+1)],
                         index=h3_profiles.index)

    frame.insert(0, "count", h3_profiles["count"])

    return frame, variance


# variance
# array([3.40877969e-01, 1.68181612e-01, 1.42295132e-01, 1.08668200e-01,
#        8.87431126e-02, 5.52068520e-02, 4.39729173e-02, 3.35064925e-02,
#        1.77240765e-02, 8.23636328e-04, 4.56495503e-20])

# coords
#                   PC1     PC2     PC3   ...
# 861188017ffffff  -2.31    0.84   -0.12
# 86118806fffffff   1.47   -1.20    0.55
# 8611880a7ffffff   0.09    2.03   -1.31