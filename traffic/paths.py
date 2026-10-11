"""
Every file location in the project, anchored at the project root so modules run the
same from any working directory.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------------------ inputs
DATA = ROOT / "data"
WEATHER_CSV = DATA / "weather" / "pachmarhi_daily_2015_2025.csv"
EXTREME_EVENTS_CSV = DATA / "weather" / "pachmarhi_extreme_events.csv"
WEATHER_METADATA_CSV = DATA / "weather" / "metadata_sheet.csv"
VEHICLES_CSV = DATA / "vehicles" / "army_vehicles_attributes.csv"
TERRAIN = DATA / "terrain"
SEASONS = ("monsoon", "summer", "winter")


def terrain_stack(season: str) -> Path:
    """17-band GeoTIFF, 30 m, EPSG:32643 (GEE export)."""
    return TERRAIN / f"pachmarhi_{season}_feature_stack.tif"


def terrain_table(season: str) -> Path:
    """Same pixels as the GeoTIFF flattened to a table, without coordinates."""
    return TERRAIN / f"pachmarhi_{season}_feature_table.parquet"


# ----------------------------------------------------------------------- outputs
OUTPUTS = ROOT / "outputs"
FORECAST_OUT = OUTPUTS / "forecasting"      # model comparison: run, report, artifact
FORECASTER_OUT = OUTPUTS / "forecaster"     # the chosen GRU + Ridge, margins, decisions
EDA_OUT = OUTPUTS / "eda"
VEHICLES_OUT = OUTPUTS / "vehicles"
VEHICLES_VCI_CSV = VEHICLES_OUT / "vehicles_vci.csv"
