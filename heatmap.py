import folium
import pandas as pd
import numpy as np
from folium.plugins import HeatMap

# NC county centroids (lat, lon) for the 90 Cornwell & Trumbull counties.
# Approximated from NC county FIPS ordering — good enough for a heatmap demo.
# In production, join to a proper shapefile (e.g. census TIGER).
NC_COUNTY_COORDS = {
    1:  (35.78, -78.64),   2:  (36.35, -76.22),   3:  (35.42, -77.05),
    4:  (35.04, -76.88),   5:  (36.55, -79.77),    6:  (35.38, -82.32),
    7:  (35.73, -81.34),   8:  (35.24, -80.84),    9:  (34.00, -78.18),
    10: (35.01, -77.32),   11: (35.32, -79.09),    12: (36.49, -77.67),
    13: (36.39, -78.55),   14: (34.97, -79.43),    15: (35.54, -78.37),
    16: (35.82, -80.19),   17: (35.31, -80.48),    18: (36.37, -80.85),
    19: (35.97, -77.79),   20: (35.49, -79.68),    21: (35.80, -78.10),
    22: (36.07, -80.63),   23: (35.01, -76.29),    24: (36.07, -76.69),
    25: (35.54, -77.39),   26: (35.71, -82.99),    27: (35.01, -83.38),
    28: (36.38, -79.39),   29: (35.70, -79.26),    30: (35.66, -80.48),
    31: (35.20, -78.60),   32: (36.55, -78.22),    33: (35.53, -80.86),
    34: (35.64, -82.48),   35: (36.47, -76.76),    36: (35.18, -82.74),
    37: (36.23, -79.97),   38: (36.43, -81.47),    39: (34.94, -80.02),
    40: (35.84, -81.00),   41: (35.74, -78.84),    42: (34.89, -77.80),
    43: (36.12, -79.35),   44: (35.85, -79.10),    45: (35.09, -81.15),
    46: (35.44, -78.93),   47: (36.55, -81.07),    48: (36.05, -77.23),
    49: (35.59, -81.68),   50: (35.47, -82.87),    51: (36.40, -77.12),
    52: (36.23, -77.82),   53: (35.22, -77.62),    54: (35.19, -81.53),
    55: (34.72, -76.64),   56: (36.45, -78.91),    57: (34.57, -77.38),
    58: (35.15, -79.97),   59: (35.36, -78.18),    60: (36.55, -80.23),
    61: (36.22, -81.82),   62: (35.06, -80.36),    63: (35.96, -80.29),
    64: (35.73, -77.05),   65: (35.99, -76.75),    66: (35.59, -78.64),
    67: (35.19, -80.12),   68: (36.30, -78.04),    69: (36.40, -80.47),
    70: (35.04, -78.21),   71: (35.35, -77.91),    72: (35.87, -76.87),
    73: (36.25, -76.94),   74: (36.14, -77.55),    75: (35.63, -77.79),
    76: (35.22, -79.23),   77: (36.36, -80.06),    78: (36.45, -79.13),
    79: (35.93, -81.54),   80: (36.18, -78.67),    81: (35.39, -81.98),
    82: (36.56, -79.55),   83: (35.56, -79.42),    84: (35.73, -83.41),
    85: (35.88, -78.55),   86: (35.28, -83.19),    87: (35.07, -79.75),
    88: (35.69, -78.25),   89: (34.83, -79.18),    90: (36.07, -78.36),
}


def build_heatmap(df: pd.DataFrame, year_filter: int = None) -> folium.Map:
    """
    Build a Folium heatmap of NC crime rates.
    Optionally filter to a specific year.
    Returns a folium.Map object.
    """
    data = df.copy()
    if year_filter:
        data = data[data["year"] == year_filter]

    # Aggregate to county mean crime rate for the selected period
    county_crime = (
        data.groupby("county")["crmrte"]
        .mean()
        .reset_index()
        .rename(columns={"crmrte": "mean_crmrte"})
    )

    # Attach coordinates
    county_crime["lat"] = county_crime["county"].map(
        lambda c: NC_COUNTY_COORDS.get(c, (35.5, -79.5))[0]
    )
    county_crime["lon"] = county_crime["county"].map(
        lambda c: NC_COUNTY_COORDS.get(c, (35.5, -79.5))[1]
    )

    # Normalise crime rates to 0-1 for heat intensity
    mn = county_crime["mean_crmrte"].min()
    mx = county_crime["mean_crmrte"].max()
    county_crime["intensity"] = (county_crime["mean_crmrte"] - mn) / (mx - mn + 1e-9)

    # Base map centred on NC
    m = folium.Map(
        location=[35.5, -79.5],
        zoom_start=7,
        tiles="CartoDB positron",
    )

    # HeatMap layer
    heat_data = [
        [row["lat"], row["lon"], row["intensity"]]
        for _, row in county_crime.iterrows()
    ]
    HeatMap(
        heat_data,
        radius=25,
        blur=20,
        max_zoom=10,
        gradient={0.2: "blue", 0.5: "lime", 0.8: "orange", 1.0: "red"},
    ).add_to(m)

    # Circle markers with tooltips
    for _, row in county_crime.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=6,
            color="white",
            weight=1,
            fill=True,
            fill_color=_crime_color(row["intensity"]),
            fill_opacity=0.8,
            tooltip=folium.Tooltip(
                f"<b>County {int(row['county'])}</b><br>"
                f"Crime rate: {row['mean_crmrte']:.4f}<br>"
                f"Intensity: {row['intensity']:.2f}"
            ),
        ).add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:10px 16px;border-radius:8px;
                border:1px solid #ccc;font-size:12px;font-family:sans-serif">
        <b>Crime Rate Intensity</b><br>
        <span style="color:blue">&#9632;</span> Low &nbsp;
        <span style="color:lime">&#9632;</span> Medium &nbsp;
        <span style="color:orange">&#9632;</span> High &nbsp;
        <span style="color:red">&#9632;</span> Highest
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


def _crime_color(intensity: float) -> str:
    if intensity < 0.25:
        return "#3B8BD4"
    elif intensity < 0.5:
        return "#63C463"
    elif intensity < 0.75:
        return "#EF9F27"
    else:
        return "#E24B4A"
