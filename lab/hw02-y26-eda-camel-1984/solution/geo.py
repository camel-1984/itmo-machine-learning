import math
from functools import lru_cache
from pathlib import Path
import geopandas as gpd
import h3
import pandas as pd

BOUNDARY_PATH = Path(__file__).resolve().parent / "assets" / "vladimir_boundary.geojson"


@lru_cache(maxsize=1)
def load_boundary(path: str = str(BOUNDARY_PATH)):
    frame = gpd.read_file(path)
    if frame.crs is None:
        frame = frame.set_crs("EPSG:4326")
    return frame.to_crs("EPSG:4326").geometry.union_all()

def edge_km(resolution: int) -> float:
    return round(h3.average_hexagon_edge_length(resolution, unit="km"), 2)

def region_mask(frame: pd.DataFrame, lat: str = "latitude", lon: str = "longitude") -> pd.Series:
    boundary = load_boundary()
    min_lon, min_lat, max_lon, max_lat = boundary.bounds

    latitude = pd.to_numeric(frame[lat], errors="coerce")
    longitude = pd.to_numeric(frame[lon], errors="coerce")
    inside_box = latitude.between(min_lat, max_lat) & longitude.between(min_lon, max_lon)

    mask = pd.Series(False, index=frame.index)
    if not inside_box.any():
        return mask

    candidates = frame.index[inside_box]
    points = gpd.GeoSeries(
        gpd.points_from_xy(longitude[candidates], latitude[candidates]),
        index=candidates, crs="EPSG:4326",
    )
    mask.loc[candidates] = points.within(boundary)
    return mask

def no_coords_mask(frame: pd.DataFrame, lat: str = "latitude", lon: str = "longitude") -> pd.Series:
    latitude = pd.to_numeric(frame[lat], errors="coerce")
    longitude = pd.to_numeric(frame[lon], errors="coerce")
    return latitude.isna() | longitude.isna()

def coverage(frame: pd.DataFrame, mask: pd.Series) -> dict[str, int]:
    no_coords = no_coords_mask(frame)
    return {
        "total": len(frame),
        "shown": int(mask.sum()),
        "no_coords": int(no_coords.sum()),
        "outside": int((~no_coords & ~mask).sum()),
    }

def cell_ids(
    frame: pd.DataFrame,
    resolution: int = 7,
    lat: str = "latitude",
    lon: str = "longitude",
) -> pd.Series:
    cells = [
        h3.latlng_to_cell(float(y), float(x), resolution)
        for y, x in zip(frame[lat], frame[lon])
    ]
    return pd.Series(cells, index=frame.index, name="cell")


def hex_counts(cells: pd.Series) -> pd.DataFrame:
    return cells.value_counts().rename("count").rename_axis("cell").reset_index()


@lru_cache(maxsize=20000)
def cell_ring(cell: str) -> tuple:
    return tuple((lng, lat) for lat, lng in h3.cell_to_boundary(cell))


def hex_geojson(cells) -> dict:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": cell,
                "properties": {"cell": cell},
                "geometry": {"type": "Polygon", "coordinates": [list(cell_ring(cell))]},
            }
            for cell in cells
        ],
    }


def hex_frame(
    frame: pd.DataFrame,
    resolution: int = 7,
    lat: str = "latitude",
    lon: str = "longitude",
) -> tuple[pd.DataFrame, dict]:
    counts = hex_counts(cell_ids(frame, resolution, lat, lon))
    return counts, hex_geojson(counts["cell"])


def zoom_for(
    frame: pd.DataFrame,
    lat: str = "latitude",
    lon: str = "longitude",
    width_factor: float = 1.4,
) -> float:
    latitude = pd.to_numeric(frame[lat], errors="coerce").dropna()
    longitude = pd.to_numeric(frame[lon], errors="coerce").dropna()
    if latitude.empty or longitude.empty:
        return 6.0

    span = max(
        float(longitude.max() - longitude.min()),
        float(latitude.max() - latitude.min()) * 1.6,
        0.01,
    )
    return float(min(13.0, max(4.0, math.log2(360 / span) + width_factor)))
