import math

import h3
import numpy as np
import pandas as pd
import geo
import data_loading

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE



def h3_to_pca(
    df: pd.DataFrame,
    resolution: int = 6,
    min_count: int = 30,
    top: int = 4,
) -> pd.DataFrame:

    frame = df if "latitude" in df.columns else data_loading.full_frame_to_app(df)
    frame = frame[geo.region_mask(frame)]
    frame["cell"] = [
        h3.latlng_to_cell(lat, lon, resolution)
        for lat, lon in zip(frame["latitude"], frame["longitude"])
    ]

    counts = frame["cell"].value_counts()
    frame = frame[frame["cell"].isin(counts[counts >= min_count].index)]

    blocks = []
    for column in ("category", "severity", "light"):
        shares = pd.crosstab(frame["cell"], frame[column], normalize="index")
        blocks.append(shares[shares.sum().nlargest(top).index].add_prefix(f"{column}_"))

    h3_profiles = pd.concat(blocks, axis="columns").fillna(0.0)
    h3_profiles.insert(0, "count", counts)

    return h3_profiles

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

def analise_weights(
        df: pd.DataFrame,
        pc_count: int = 2
) -> dict[str, pd.Series]:

    significant = {}
    threshold = 1/math.sqrt(len(df))

    for i in range(1, pc_count+1):
        s = df[f"pc{i}"]
        significant[f"pc{i}"] = s[s.abs() > threshold].sort_values(

        )
    return significant



def variance_frame(
        variance: np.ndarray
) -> pd.DataFrame:

    frame = pd.DataFrame(index=[f"PC{i}" for i in range(1, len(variance)+1)])
    frame["variance"] = variance
    frame["cumulative"] = variance.cumsum()

    return frame


def pc_edges(
        h3_profiles: pd.DataFrame,
        coords: pd.DataFrame,
        component: str = "pc1",
        n: int = 10,
) -> pd.DataFrame:

    axis = coords[component]
    features = h3_profiles.drop(columns="count")

    low = features.loc[axis.nsmallest(n).index].mean()
    high = features.loc[axis.nlargest(n).index].mean()

    frame = pd.DataFrame({"low": low, "high": high})
    frame["diff"] = frame["high"] - frame["low"]

    return frame.sort_values("diff")


def tsne(
        h3_profiles: pd.DataFrame,
        perplexity: float = 30.0,
        random_state: int = 0,
        max_iter: int = 1000,
) -> pd.DataFrame:

    x = h3_profiles.drop(columns="count")
    x_scaled = StandardScaler().fit_transform(x)

    model = TSNE(
        n_components=2, perplexity=perplexity, random_state=random_state,
        max_iter=max_iter, init="pca",
    )
    coords = model.fit_transform(x_scaled)

    frame = pd.DataFrame(coords, columns=["tsne1", "tsne2"], index=h3_profiles.index)
    frame.insert(0, "count", h3_profiles["count"])

    return frame
