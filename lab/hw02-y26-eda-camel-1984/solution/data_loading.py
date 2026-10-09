import json
from pathlib import Path
import pandas as pd
import utils


def load_records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict) and payload.get("type") == "FeatureCollection":
        records = payload.get("features")
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError("Ожидается GeoJSON FeatureCollection или список записей.")
    if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
        raise ValueError("Выгрузка должна содержать список объектов JSON.")
    return records


def properties(record: dict) -> dict:
    if record.get("type") == "Feature":
        value = record.get("properties")
        return value if isinstance(value, dict) else {}
    return record


def record_to_full_frame(records: list[dict]) -> pd.DataFrame:
    rows = []
    for index, record in enumerate(records):
        props = properties(record)
        row = props.copy()

        row["source_id"] = index

        geometry = record.get("geometry")
        geometry = geometry if isinstance(geometry, dict) else {}
        coords = geometry.get("coordinates")
        if geometry.get("type") == "Point" and isinstance(coords, list) and len(coords) >= 2:
            row["geom_longitude"], row["geom_latitude"] = coords[:2]
        else:
            row["geom_longitude"], row["geom_latitude"] = None, None

        rows.append(row)

    frame = pd.DataFrame(rows)
    frame = utils.unpack_point(frame)
    return frame

def full_frame_to_app(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["source_row"] = frame["source_id"]
    frame["datetime_raw"] = frame["datetime"]
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce", format="mixed")

    for name, geom, point in (
        ("longitude", "geom_longitude", "point_longitude"),
        ("latitude", "geom_latitude", "point_latitude"),
    ):
        values = pd.to_numeric(frame[geom], errors="coerce")
        frame[name] = values.where(values.notna(), pd.to_numeric(frame[point], errors="coerce"))

    return frame
