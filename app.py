from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="GP Registration Divergence Explorer",
    page_icon="📊",
    layout="wide",
)

# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

DATA_PATH = (
    Path(__file__).parent
    / "data"
    / "list_size_census_comparison.csv"
)

GEOJSON_PATH = (
    Path(__file__).parent
    / "data"
    / "local_authority_boundaries.geojson"
)

REGION_ORDER = [
    "East Midlands",
    "East of England",
    "London",
    "North East",
    "North West",
    "South East",
    "South West",
    "West Midlands",
    "Yorkshire and The Humber",
]


# --------------------------------------------------
# LOAD AND VALIDATE DATA
# --------------------------------------------------

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={"Local Authority Code": "string"},
    )

    required = {
        "Local Authority Code",
        "Local Authority Name",
        "Census Count",
        "List Size",
        "Region Name",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    # Convert counts containing thousands separators.
    for column in ["Census Count", "List Size"]:
        df[column] = pd.to_numeric(
            df[column]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip(),
            errors="coerce",
        )

    df["Region Name"] = (
        df["Region Name"].astype("string").str.strip()
    )

    df["Local Authority Name"] = (
        df["Local Authority Name"]
        .astype("string")
        .str.strip()
    )

    # Retain rows with the information needed for analysis.
    df = df.dropna(
        subset=[
            "Census Count",
            "List Size",
            "Region Name",
            "Local Authority Name",
        ]
    ).copy()

    # Descriptive measures.
    df["Difference"] = (
        df["List Size"] - df["Census Count"]
    )

    df["Percentage Difference"] = (
        df["Difference"]
        .div(df["Census Count"].where(df["Census Count"] != 0))
        * 100
    )

    return df


# --------------------------------------------------
# FORMATTING AND SUMMARY METRICS
# --------------------------------------------------

def fmt_count(value):
    if pd.isna(value):
        return "—"

    return f"{value:,.0f}"


def fmt_pct(value):
    if pd.isna(value):
        return "—"

    return f"{value:+.2f}%"


def add_metrics(frame: pd.DataFrame):
    census = frame["Census Count"].sum()
    registrations = frame["List Size"].sum()

    difference = registrations - census

    percentage = (
        difference / census * 100
        if census != 0
        else float("nan")
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Census population",
        fmt_count(census),
    )

    col2.metric(
        "GP registrations",
        fmt_count(registrations),
    )

    col3.metric(
        "Absolute difference",
        fmt_count(difference),
    )

    col4.metric(
        "Difference vs Census",
        fmt_pct(percentage),
    )


# --------------------------------------------------
# APPLICATION HEADER
# --------------------------------------------------

st.title("GP Registration Divergence Explorer")

st.caption(
    "Descriptive comparison of GP registration counts "
    "and Census 2021 population at English local-authority "
    "level. Identify patterns to inform further investigation "
    "rather than infer causes."
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

if not DATA_PATH.exists():
    st.error(
        "Dataset not found. Place "
        "'list_size_census_comparison.csv' "
        "inside the data folder."
    )
    st.stop()

try:
    data = load_data(str(DATA_PATH))

except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()


# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------

regions_in_data = sorted(
    data["Region Name"].dropna().unique().tolist()
)

regions = [
    region
    for region in REGION_ORDER
    if region in regions_in_data
]

regions += [
    region
    for region in regions_in_data
    if region not in regions
]

with st.sidebar:

    st.header("Explore the data")

    selected_region = st.selectbox(
        "Select region",
        ["England"] + regions,
        index=0,
    )

    view = st.selectbox(
        "Analysis view",
        [
            "Regional comparison",
            "Local-authority analysis",
            "Interactive map",
        ],
    )

    measure_label = st.selectbox(
        "Comparison measure",
        [
            "Percentage difference",
            "Absolute difference",
            "GP registrations",
            "Census population",
        ],
        help=(
            "Percentage difference is useful when comparing "
            "areas with different population sizes."
        ),
    )

    st.divider()

    st.caption(
        "Source: NHS England Digital, "
        "'List Size to Census Comparison' dataset."
    )

    st.caption(
        "Reference dates: GP registrations, 1 April 2021; "
        "Census, 21 March 2021."
    )


# Map display labels to dataframe column names.
measure_map = {
    "Percentage difference": "Percentage Difference",
    "Absolute difference": "Difference",
    "GP registrations": "List Size",
    "Census population": "Census Count",
}

measure = measure_map[measure_label]


# --------------------------------------------------
# FILTER DATA
# --------------------------------------------------

if selected_region == "England":
    scope = data.copy()

else:
    scope = data[
        data["Region Name"] == selected_region
    ].copy()

if scope.empty:
    st.warning("No rows found for this selection.")
    st.stop()


# --------------------------------------------------
# VIEW 1: REGIONAL COMPARISON
# --------------------------------------------------

if view == "Regional comparison":

    st.subheader(
        "How do the nine English regions compare?"
    )

    if selected_region != "England":
        st.info(
            "This view compares all nine regions. "
            "Use Local-authority analysis to focus on "
            "your selected region."
        )

    # National headline metrics.
    add_metrics(data)

    # Aggregate local-authority counts by region.
    regional = (
        data.groupby("Region Name", as_index=False)
        .agg(
            {
                "Census Count": "sum",
                "List Size": "sum",
            }
        )
    )

    regional["Difference"] = (
        regional["List Size"]
        - regional["Census Count"]
    )

    regional["Percentage Difference"] = (
        regional["Difference"]
        / regional["Census Count"].where(
            regional["Census Count"] != 0
        )
        * 100
    )

    regional["Region Name"] = pd.Categorical(
        regional["Region Name"],
        categories=regions,
        ordered=True,
    )

    regional = regional.sort_values("Region Name")

    # Plot either population counts or divergence.
    if measure in ["List Size", "Census Count"]:

        chart_data = regional.melt(
            id_vars=["Region Name"],
            value_vars=["Census Count", "List Size"],
            var_name="Dataset",
            value_name="People",
        )

        chart_data["Dataset"] = chart_data[
            "Dataset"
        ].map(
            {
                "Census Count": "Census population",
                "List Size": "GP registrations",
            }
        )

        fig = px.bar(
            chart_data,
            x="Region Name",
            y="People",
            color="Dataset",
            barmode="group",
            title=(
                "GP registrations and Census population "
                "by region"
            ),
            labels={
                "Region Name": "Region",
                "People": "People",
            },
        )

    else:

        fig = px.bar(
            regional.sort_values(measure),
            x=measure,
            y="Region Name",
            orientation="h",
            title=f"Regional {measure_label.lower()}",
            labels={
                "Region Name": "Region",
                measure: measure_label,
            },
            hover_data={
                "Census Count": ":,.0f",
                "List Size": ":,.0f",
                "Difference": ":,.0f",
                "Percentage Difference": ":.2f",
            },
        )

    fig.update_layout(
        legend_title_text="",
        margin=dict(l=10, r=10, t=60, b=10),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # Regional results table.
    table = regional.copy()

    table["Region Name"] = (
        table["Region Name"].astype(str)
    )

    table = table.rename(
        columns={
            "Region Name": "Region",
            "Census Count": "Census population",
            "List Size": "GP registrations",
            "Difference": "Absolute difference",
            "Percentage Difference": "Difference vs Census (%)",
        }
    )

    st.dataframe(
        table[
            [
                "Region",
                "Census population",
                "GP registrations",
                "Absolute difference",
                "Difference vs Census (%)",
            ]
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Census population": st.column_config.NumberColumn(
                format="%,.0f"
            ),
            "GP registrations": st.column_config.NumberColumn(
                format="%,.0f"
            ),
            "Absolute difference": st.column_config.NumberColumn(
                format="%,.0f"
            ),
            "Difference vs Census (%)": (
                st.column_config.NumberColumn(
                    format="%.2f%%"
                )
            ),
        },
    )


# --------------------------------------------------
# VIEW 2: LOCAL-AUTHORITY ANALYSIS
# --------------------------------------------------

elif view == "Local-authority analysis":

    heading = (
        "England"
        if selected_region == "England"
        else selected_region
    )

    st.subheader(
        f"Local-authority analysis: {heading}"
    )

    add_metrics(scope)

    # Rank local authorities using the selected measure.
    ordered = scope.sort_values(
        measure,
        ascending=False,
        na_position="last",
    )

    # Display the 30 highest values.
    fig = px.bar(
        ordered.head(30),
        x=measure,
        y="Local Authority Name",
        orientation="h",
        title=(
            f"30 highest local-authority values: "
            f"{measure_label.lower()}"
        ),
        labels={
            measure: measure_label,
            "Local Authority Name": "Local authority",
        },
        hover_data={
            "Region Name": True,
            "Census Count": ":,.0f",
            "List Size": ":,.0f",
            "Difference": ":,.0f",
            "Percentage Difference": ":.2f",
        },
    )

    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        margin=dict(l=10, r=10, t=60, b=10),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # Distribution of percentage divergence.
    st.markdown(
        "#### Distribution of local-authority differences"
    )

    histogram = px.histogram(
        scope,
        x="Percentage Difference",
        nbins=25,
        title=(
            "Percentage divergence across "
            "local authorities"
        ),
        labels={
            "Percentage Difference": (
                "Difference vs Census (%)"
            ),
            "count": "Local authorities",
        },
    )

    histogram.update_layout(
        margin=dict(l=10, r=10, t=60, b=10),
    )

    st.plotly_chart(
        histogram,
        use_container_width=True,
    )

    # Local-authority detail table.
    st.markdown("#### Local-authority data")

    table = scope[
        [
            "Local Authority Code",
            "Local Authority Name",
            "Region Name",
            "Census Count",
            "List Size",
            "Difference",
            "Percentage Difference",
        ]
    ].copy()

    table = table.rename(
        columns={
            "Local Authority Code": "Authority code",
            "Local Authority Name": "Local authority",
            "Region Name": "Region",
            "Census Count": "Census population",
            "List Size": "GP registrations",
            "Difference": "Absolute difference",
            "Percentage Difference": "Difference vs Census (%)",
        }
    )

    table = table.sort_values(
        "Difference vs Census (%)",
        ascending=False,
    )

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Census population": st.column_config.NumberColumn(
                format="%,.0f"
            ),
            "GP registrations": st.column_config.NumberColumn(
                format="%,.0f"
            ),
            "Absolute difference": st.column_config.NumberColumn(
                format="%,.0f"
            ),
            "Difference vs Census (%)": (
                st.column_config.NumberColumn(
                    format="%.2f%%"
                )
            ),
        },
    )

    # Allow export of the displayed data.
    st.download_button(
        "Download displayed data as CSV",
        table.to_csv(index=False).encode("utf-8"),
        file_name="gp_registration_divergence.csv",
        mime="text/csv",
    )


# --------------------------------------------------
# VIEW 3: INTERACTIVE MAP
# --------------------------------------------------

else:

    st.subheader(
        "Interactive local-authority map"
    )

    st.write(
        "Colour local authorities by the selected measure. "
        "The map requires published boundary geometry joined "
        "using official local-authority codes."
    )

    if not GEOJSON_PATH.exists():

        st.warning(
            "The boundary GeoJSON file has not been added yet."
        )

        st.info(
            "To activate the map, add a compatible GeoJSON "
            "file at data/local_authority_boundaries.geojson. "
            "Its features must contain official local-authority "
            "codes in a recognised property such as LAD24CD "
            "or LAD23CD."
        )

        st.write(
            "Until then, use the Local-authority analysis "
            "view to explore and rank the areas."
        )

    else:

        try:
            with GEOJSON_PATH.open(
                "r",
                encoding="utf-8",
            ) as file:
                geojson = json.load(file)

            code_candidates = [
                "LAD24CD",
                "LAD23CD",
                "LAD22CD",
                "LAD21CD",
                "LAD20CD",
                "LAD19CD",
                "LAD18CD",
                "LAD17CD",
            ]

            available_properties = set()

            for feature in geojson.get("features", []):
                available_properties.update(
                    (feature.get("properties") or {}).keys()
                )

            code_property = next(
                (
                    property_name
                    for property_name in code_candidates
                    if property_name in available_properties
                ),
                None,
            )

            if not code_property:

                st.error(
                    "No recognised local-authority code "
                    "property was found in the GeoJSON."
                )

            else:

                # Only show local authorities in the selected scope.
                fig = px.choropleth(
                    scope,
                    geojson=geojson,
                    locations="Local Authority Code",
                    featureidkey=(
                        f"properties.{code_property}"
                    ),
                    color=measure,
                    hover_name="Local Authority Name",
                    hover_data={
                        "Region Name": True,
                        "Census Count": ":,.0f",
                        "List Size": ":,.0f",
                        "Difference": ":,.0f",
                        "Percentage Difference": ":.2f",
                    },
                    labels={
                        measure: measure_label,
                    },
                    title=(
                        f"{measure_label} by local authority"
                    ),
                )

                fig.update_geos(
                    fitbounds="locations",
                    visible=False,
                )

                fig.update_layout(
                    margin=dict(l=0, r=0, t=60, b=0),
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

        except Exception as exc:

            st.error(
                f"Could not display the map: {exc}"
            )


# --------------------------------------------------
# INTERPRETATION AND LIMITATIONS
# --------------------------------------------------

st.divider()

st.markdown(
    "### Interpretation and limitations"
)

st.warning(
    "A difference between GP registrations and Census "
    "population is not a direct count of incorrect, duplicate "
    "or 'ghost' patient records. The datasets have different "
    "purposes and reference dates. This dashboard is descriptive: "
    "it identifies patterns and outliers to guide further "
    "investigation, but it does not establish their causes."
)

st.caption(
    "Method: Difference = GP registrations − Census population. "
    "Percentage difference = Difference ÷ Census population × 100. "
    "Regional and national figures are calculated by summing "
    "the available local-authority rows."
)
