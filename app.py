import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Bund Future - Volatilite roulante", layout="wide")


@st.cache_data
def load_excel(file):
    return pd.read_excel(file)


def heure_to_timedelta(value):
    """Convertit une cellule de la colonne Heure (str, datetime.time, ou fraction Excel) en Timedelta."""
    if pd.isna(value):
        return pd.Timedelta(0)
    if isinstance(value, str):
        return pd.to_timedelta(value)
    if hasattr(value, "hour"):
        return pd.Timedelta(hours=value.hour, minutes=value.minute, seconds=value.second)
    return pd.to_timedelta(float(value), unit="D")


def build_datetime(df, date_col, heure_col):
    dates = pd.to_datetime(df[date_col]).dt.normalize()
    deltas = df[heure_col].apply(heure_to_timedelta)
    return dates + deltas


st.title("Dashboard - Volatilite roulante des futures Bund")

# ----------------------------------------------------------------------------
# 1. Chargement des fichiers
# ----------------------------------------------------------------------------
st.sidebar.header("1. Fichiers")
events_file = st.sidebar.file_uploader("Fichier evenements (.xlsx)", type=["xlsx"])
bund_file = st.sidebar.file_uploader("Fichier b_.xlsx", type=["xlsx"])

if not events_file or not bund_file:
    st.info("Charge les deux fichiers Excel dans la barre laterale pour demarrer.")
    st.stop()

df_events_raw = load_excel(events_file)
df_bund_raw = load_excel(bund_file)

# ----------------------------------------------------------------------------
# 2. Mapping des colonnes (au cas ou les noms different d'un fichier a l'autre)
# ----------------------------------------------------------------------------
st.sidebar.header("2. Colonnes - fichier evenements")
event_col = st.sidebar.selectbox("Colonne Event", df_events_raw.columns)
date_col_events = st.sidebar.selectbox("Colonne Date", df_events_raw.columns)
relevance_options = ["(aucune)"] + list(df_events_raw.columns)
relevance_col = st.sidebar.selectbox("Colonne Relevance", relevance_options)

st.sidebar.header("3. Colonnes - .xlsx")
bund_columns = list(df_bund_raw.columns)
date_col_bund = st.sidebar.selectbox("Colonne Date", bund_columns, index=min(1, len(bund_columns) - 1))
heure_col_bund = st.sidebar.selectbox("Colonne Heure", bund_columns, index=min(2, len(bund_columns) - 1))
price_col_bund = st.sidebar.selectbox("Colonne Prix future (4eme colonne)", bund_columns, index=min(3, len(bund_columns) - 1))

st.sidebar.header("4. Parametres volatilite")
window_minutes = st.sidebar.number_input("Fenetre de calcul (minutes)", min_value=1, value=15, step=1)
vol_method = st.sidebar.radio("Methode", ["Ecart-type des rendements", "Ecart-type du prix brut"])

# ----------------------------------------------------------------------------
# 3. Selection de l'evenement puis de la date
# ----------------------------------------------------------------------------
df_events = df_events_raw.copy()
df_events[date_col_events] = pd.to_datetime(df_events[date_col_events])

events_list = sorted(df_events[event_col].dropna().unique())
selected_event = st.selectbox("Evenement", events_list)

df_event_filtered = df_events[df_events[event_col] == selected_event].sort_values(date_col_events)

available_dates = sorted(df_event_filtered[date_col_events].dt.date.unique())
if not available_dates:
    st.warning("Aucune date disponible pour cet evenement.")
    st.stop()

selected_date = st.selectbox("Date", available_dates, format_func=lambda d: d.strftime("%d/%m/%Y"))

# ----------------------------------------------------------------------------
# 4. Filtrage du fichier
# ----------------------------------------------------------------------------
df_bund = df_bund_raw.copy()
df_bund["__datetime__"] = build_datetime(df_bund, date_col_bund, heure_col_bund)
df_day = df_bund[df_bund["__datetime__"].dt.date == selected_date].sort_values("__datetime__")

if df_day.empty:
    st.warning("Aucune donnee Bund trouvee pour cette date.")
    st.stop()

df_day = df_day.set_index("__datetime__")
df_day[price_col_bund] = pd.to_numeric(df_day[price_col_bund], errors="coerce")

# ----------------------------------------------------------------------------
# 5. Calcul de la volatilite roulante (fenetre temporelle, pas un nombre de lignes,
#    pour rester correct meme si les heures ne sont pas parfaitement equidistantes)
# ----------------------------------------------------------------------------
if vol_method == "Ecart-type des rendements":
    series_for_vol = df_day[price_col_bund].pct_change()
else:
    series_for_vol = df_day[price_col_bund]

df_day["rolling_vol"] = series_for_vol.rolling(f"{window_minutes}min").std()

# ----------------------------------------------------------------------------
# 6. Affichage : graphe + petit tableau de relevance a cote
# ----------------------------------------------------------------------------
col_graph, col_table = st.columns([4, 1])

with col_graph:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_day.index, y=df_day["rolling_vol"], mode="lines", name="Volatilite roulante"))
    fig.update_layout(
        title=f"Volatilite roulante ({window_minutes} min) - {selected_event} - {selected_date.strftime('%d/%m/%Y')}",
        xaxis_title="Heure",
        yaxis_title="Volatilite",
        height=520,
    )
    st.plotly_chart(fig, use_container_width=True)

with col_table:
    st.subheader("Relevance")
    if relevance_col != "(aucune)":
        relevance_mean_event = df_event_filtered[relevance_col].mean()
        row_for_date = df_event_filtered[df_event_filtered[date_col_events].dt.date == selected_date]
        relevance_today = row_for_date[relevance_col].mean() if not row_for_date.empty else np.nan
        st.table(pd.DataFrame(
            {"Valeur": [relevance_today, relevance_mean_event]},
            index=["Date selectionnee", "Moyenne (evenement)"],
        ))
    else:
        st.caption("Choisis une colonne Relevance dans la barre laterale pour l'afficher ici.")

with st.expander("Donnees brutes du jour (Bund)"):
    st.dataframe(df_day[[price_col_bund, "rolling_vol"]])

with st.expander("Donnees de l'evenement selectionne"):
    st.dataframe(df_event_filtered)
