import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re
import io

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ML en Clasificación de Frutas",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  /* Google Fonts */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Playfair+Display:wght@700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0D1117;
    color: #E6EDF3;
  }

  /* Hero header */
  .hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #58A6FF;
    line-height: 1.2;
    margin-bottom: 0.2rem;
  }
  .hero-sub {
    font-size: 1rem;
    color: #8B949E;
    margin-bottom: 1.5rem;
  }
  .keyword-pill {
    display: inline-block;
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 20px;
    padding: 4px 14px;
    margin: 4px;
    font-size: 0.82rem;
    color: #58A6FF;
    font-weight: 600;
  }

  /* Metric cards */
  .metric-card {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
  }
  .metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #3FB950;
  }
  .metric-label {
    font-size: 0.82rem;
    color: #8B949E;
    margin-top: 4px;
  }

  /* Section headers */
  .section-header {
    font-size: 1.15rem;
    font-weight: 600;
    color: #E6EDF3;
    border-left: 3px solid #58A6FF;
    padding-left: 12px;
    margin: 32px 0 16px 0;
  }

  /* Insight box */
  .insight-box {
    background: #161B22;
    border: 1px solid #30363D;
    border-left: 3px solid #3FB950;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 0.88rem;
    color: #8B949E;
    margin-top: 8px;
  }

  /* Upload area */
  .upload-hint {
    background: #161B22;
    border: 1px dashed #30363D;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    color: #8B949E;
    font-size: 0.88rem;
    margin-bottom: 16px;
  }

  /* Streamlit overrides */
  .stDataFrame { border-radius: 10px; }
  div[data-testid="stSidebar"] { background-color: #0D1117; border-right: 1px solid #21262D; }
  .stPlotlyChart { border-radius: 10px; overflow: hidden; }
  hr { border-color: #21262D; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
STOPWORDS_ES_EN = {
    "the","a","an","and","or","of","in","to","for","with","on","is","are","was","were",
    "this","that","these","those","from","by","as","at","be","been","being","have",
    "has","had","do","does","did","not","but","it","its","into","than","more","also",
    "which","that","can","will","may","their","our","we","they","i","you","he","she",
    "de","la","el","en","se","que","con","por","para","una","los","las","del","al",
    "es","un","su","no","lo","le","y","o","a","study","based","using","results",
    "data","paper","proposed","approach","method","methods","model","used","use",
    "two","three","high","new","different","however","thus","show","shows","showed",
    "analysis","significant","number","among","between","within","during","after",
    "classification","fruit","fruits","machine","learning","nutrition","nutritional",
    "food","foods","plant","plants","agricultural","agriculture","dataset","datasets",
}

PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "#161B22",
        "plot_bgcolor": "#161B22",
        "font": {"color": "#E6EDF3", "family": "Inter"},
        "xaxis": {"gridcolor": "#21262D", "linecolor": "#30363D"},
        "yaxis": {"gridcolor": "#21262D", "linecolor": "#30363D"},
        "colorway": ["#58A6FF","#3FB950","#F78166","#D2A8FF","#FFA657","#79C0FF"],
    }
}

def clean_text(text: str) -> list[str]:
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    return [w for w in words if w not in STOPWORDS_ES_EN]

def parse_authors(cell) -> list[str]:
    if pd.isna(cell) or str(cell).strip() == "":
        return []
    return [a.strip() for a in re.split(r'[;,]', str(cell)) if a.strip()]

def load_scopus(file) -> pd.DataFrame:
    df = pd.read_csv(file, encoding="utf-8-sig")
    # Normalize column names: strip spaces, unify casing
    df.columns = df.columns.str.strip()
    rename_map = {}
    for col in df.columns:
        low = col.lower()
        if "author" in low and "keyword" not in low and "id" not in low:
            rename_map[col] = "Authors"
        elif low in ("title","document title"):
            rename_map[col] = "Title"
        elif low == "year":
            rename_map[col] = "Year"
        elif "abstract" in low:
            rename_map[col] = "Abstract"
        elif "cited" in low:
            rename_map[col] = "Cited by"
        elif "source" in low:
            rename_map[col] = "Source"
    df.rename(columns=rename_map, inplace=True)

    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    if "Cited by" in df.columns:
        df["Cited by"] = pd.to_numeric(df["Cited by"], errors="coerce").fillna(0).astype(int)

    return df

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🍎 Panel de control")
    st.markdown("---")

    st.markdown("**Cargar dataset Scopus**")
    st.markdown(
        '<div class="upload-hint">Exporta tu búsqueda desde Scopus como CSV y súbela aquí.</div>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader("Archivo CSV de Scopus", type=["csv"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Filtros**")

    year_filter = None
    min_cites = 0

    if uploaded_file:
        df_raw = load_scopus(uploaded_file)
        if "Year" in df_raw.columns:
            years_valid = df_raw["Year"].dropna().astype(int)
            if not years_valid.empty:
                yr_min, yr_max = int(years_valid.min()), int(years_valid.max())
                year_filter = st.slider("Rango de años", yr_min, yr_max, (yr_min, yr_max))
        if "Cited by" in df_raw.columns:
            min_cites = st.slider("Mínimo de citas", 0, int(df_raw["Cited by"].max()), 0)

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem;color:#8B949E;'>Grupo 6 · ISIL · 2025<br>Pregunta de investigación PA3</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
#  HERO HEADER
# ─────────────────────────────────────────────
st.markdown(
    '<div class="hero-title">Machine Learning en Clasificación de Frutas</div>'
    '<div class="hero-sub">Análisis bibliométrico · Scopus · Decisiones alimentarias saludables</div>',
    unsafe_allow_html=True,
)

for kw in ["Machine Learning", "Fruits", "Classification", "Nutrition"]:
    st.markdown(f'<span class="keyword-pill">{kw}</span>', unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
#  RESEARCH QUESTION CARD
# ─────────────────────────────────────────────
st.markdown("""
<div style='background:#161B22;border:1px solid #30363D;border-radius:12px;padding:18px 22px;margin-bottom:24px;'>
  <div style='font-size:0.72rem;color:#58A6FF;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:6px;'>Pregunta de Investigación</div>
  <div style='font-size:1.05rem;color:#E6EDF3;font-weight:400;line-height:1.55;'>
    ¿Cómo se aplica el Machine Learning en la clasificación de frutas utilizando
    información nutricional para apoyar la toma de decisiones alimentarias saludables?
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MAIN CONTENT
# ─────────────────────────────────────────────
if uploaded_file is None:
    st.info("⬅️  Carga tu archivo CSV de Scopus en el panel izquierdo para visualizar el análisis.")
    st.stop()

# Apply filters
df = df_raw.copy()
if year_filter and "Year" in df.columns:
    df = df[(df["Year"] >= year_filter[0]) & (df["Year"] <= year_filter[1])]
if "Cited by" in df.columns:
    df = df[df["Cited by"] >= min_cites]

if df.empty:
    st.warning("No hay artículos con los filtros actuales. Ajusta el panel lateral.")
    st.stop()

# ─────────────────────────────────────────────
#  KPI METRICS
# ─────────────────────────────────────────────
total_articles = len(df)
total_cites = int(df["Cited by"].sum()) if "Cited by" in df.columns else "N/A"
unique_authors = len(set(
    a for cell in df.get("Authors", pd.Series(dtype=str)).fillna("")
    for a in parse_authors(cell)
)) if "Authors" in df.columns else "N/A"
year_span = f"{int(df['Year'].min())}–{int(df['Year'].max())}" if "Year" in df.columns and not df["Year"].isna().all() else "N/A"

c1, c2, c3, c4 = st.columns(4)
for col, val, label in zip(
    [c1, c2, c3, c4],
    [total_articles, total_cites, unique_authors, year_span],
    ["Artículos", "Citas totales", "Autores únicos", "Período"],
):
    col.markdown(
        f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("")

# ─────────────────────────────────────────────
#  ROW 1: Publications per year + Top journals
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Tendencia de publicaciones</div>', unsafe_allow_html=True)
col_a, col_b = st.columns([3, 2])

with col_a:
    if "Year" in df.columns:
        by_year = df.groupby("Year").size().reset_index(name="Artículos")
        fig_year = px.bar(
            by_year, x="Year", y="Artículos",
            title="Publicaciones por año",
            color="Artículos",
            color_continuous_scale=["#1C2B3A","#58A6FF"],
            template=PLOTLY_TEMPLATE,
        )
        fig_year.update_layout(showlegend=False, coloraxis_showscale=False,
                               title_font_size=13, margin=dict(t=40,b=20,l=10,r=10))
        fig_year.update_traces(marker_line_width=0)
        st.plotly_chart(fig_year, use_container_width=True)
        st.markdown(
            '<div class="insight-box">💡 Un aumento sostenido indica que el área está en crecimiento activo. Picos recientes sugieren alta relevancia del tema.</div>',
            unsafe_allow_html=True,
        )

with col_b:
    if "Source" in df.columns:
        top_sources = df["Source"].value_counts().head(8).reset_index()
        top_sources.columns = ["Revista", "Artículos"]
        fig_src = px.bar(
            top_sources, x="Artículos", y="Revista",
            orientation="h",
            title="Top revistas / fuentes",
            color="Artículos",
            color_continuous_scale=["#1C2B3A","#3FB950"],
            template=PLOTLY_TEMPLATE,
        )
        fig_src.update_layout(showlegend=False, coloraxis_showscale=False,
                              yaxis_title="", title_font_size=13,
                              margin=dict(t=40,b=20,l=10,r=10))
        fig_src.update_traces(marker_line_width=0)
        st.plotly_chart(fig_src, use_container_width=True)

# ─────────────────────────────────────────────
#  ROW 2: Top cited + Author network
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Impacto y autoría</div>', unsafe_allow_html=True)
col_c, col_d = st.columns(2)

with col_c:
    if "Cited by" in df.columns and "Title" in df.columns:
        top_cited = df.nlargest(10, "Cited by")[["Title","Cited by","Year"]].copy()
        top_cited["Title_short"] = top_cited["Title"].str[:55] + "…"
        fig_cites = px.bar(
            top_cited, x="Cited by", y="Title_short",
            orientation="h",
            title="Top 10 artículos más citados",
            color="Cited by",
            color_continuous_scale=["#2D1B45","#D2A8FF"],
            template=PLOTLY_TEMPLATE,
            hover_data={"Title": True, "Year": True, "Title_short": False},
        )
        fig_cites.update_layout(yaxis_title="", title_font_size=13,
                                coloraxis_showscale=False,
                                margin=dict(t=40,b=20,l=10,r=10))
        fig_cites.update_traces(marker_line_width=0)
        st.plotly_chart(fig_cites, use_container_width=True)
        st.markdown(
            '<div class="insight-box">💡 Los artículos con más citas son los más influyentes en el campo. Úsalos como referencias clave en tu marco teórico.</div>',
            unsafe_allow_html=True,
        )

with col_d:
    if "Authors" in df.columns:
        all_authors = []
        for cell in df["Authors"].fillna(""):
            all_authors.extend(parse_authors(cell))
        author_counts = Counter(all_authors).most_common(15)
        if author_counts:
            auth_df = pd.DataFrame(author_counts, columns=["Autor","Artículos"])
            fig_auth = px.bar(
                auth_df, x="Artículos", y="Autor",
                orientation="h",
                title="Autores más productivos",
                color="Artículos",
                color_continuous_scale=["#1A2E1A","#3FB950"],
                template=PLOTLY_TEMPLATE,
            )
            fig_auth.update_layout(yaxis_title="", title_font_size=13,
                                   coloraxis_showscale=False,
                                   margin=dict(t=40,b=20,l=10,r=10))
            fig_auth.update_traces(marker_line_width=0)
            st.plotly_chart(fig_auth, use_container_width=True)

# ─────────────────────────────────────────────
#  ROW 3: Word cloud + Cites distribution
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Análisis de contenido</div>', unsafe_allow_html=True)
col_e, col_f = st.columns([3, 2])

with col_e:
    if "Abstract" in df.columns:
        all_text = " ".join(df["Abstract"].dropna().astype(str))
        words = clean_text(all_text)
        if words:
            freq = Counter(words)
            wc = WordCloud(
                width=800, height=380,
                background_color="#161B22",
                colormap="Blues",
                max_words=80,
                prefer_horizontal=0.8,
                min_font_size=10,
            ).generate_from_frequencies(freq)
            fig_wc, ax = plt.subplots(figsize=(8, 3.8))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            fig_wc.patch.set_facecolor("#161B22")
            st.markdown("**Palabras clave más frecuentes en los abstracts**")
            st.pyplot(fig_wc, use_container_width=True)
            plt.close(fig_wc)
            st.markdown(
                '<div class="insight-box">💡 Las palabras dominantes reflejan los conceptos centrales del corpus. Su concentración valida la pertinencia de tus keywords de búsqueda.</div>',
                unsafe_allow_html=True,
            )

with col_f:
    if "Cited by" in df.columns:
        fig_hist = px.histogram(
            df, x="Cited by",
            nbins=20,
            title="Distribución de citas",
            template=PLOTLY_TEMPLATE,
            color_discrete_sequence=["#58A6FF"],
        )
        fig_hist.update_layout(
            xaxis_title="Número de citas",
            yaxis_title="Artículos",
            title_font_size=13,
            margin=dict(t=40,b=20,l=10,r=10),
            bargap=0.1,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Pie: cites brackets
        bins = [0, 5, 20, 50, 200, 99999]
        labels = ["0–5", "6–20", "21–50", "51–200", "200+"]
        df["cite_bracket"] = pd.cut(df["Cited by"], bins=bins, labels=labels, right=True)
        bracket_counts = df["cite_bracket"].value_counts().sort_index().reset_index()
        bracket_counts.columns = ["Rango","Count"]
        fig_pie = px.pie(
            bracket_counts, names="Rango", values="Count",
            title="Impacto por rango de citas",
            template=PLOTLY_TEMPLATE,
            color_discrete_sequence=["#58A6FF","#3FB950","#D2A8FF","#FFA657","#F78166"],
            hole=0.45,
        )
        fig_pie.update_layout(title_font_size=13, margin=dict(t=40,b=10,l=10,r=10))
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)

# ─────────────────────────────────────────────
#  ROW 4: Full dataset table
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Dataset completo</div>', unsafe_allow_html=True)

show_cols = [c for c in ["Title","Authors","Year","Source","Cited by"] if c in df.columns]
search_term = st.text_input("🔍 Buscar en títulos…", placeholder="ej. deep learning, SVM, banana…")

display_df = df[show_cols].copy()
if search_term and "Title" in display_df.columns:
    display_df = display_df[display_df["Title"].str.contains(search_term, case=False, na=False)]

st.dataframe(display_df, use_container_width=True, height=320)

# Download button
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️  Descargar dataset filtrado (CSV)",
    data=csv_bytes,
    file_name="scopus_frutas_ml_filtrado.csv",
    mime="text/csv",
)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;font-size:0.78rem;color:#8B949E;padding:10px 0;'>"
    "Grupo 6 · PA3 · ISIL 2025 · Dataset: Scopus · "
    "Keywords: Machine Learning · Fruits · Classification · Nutrition"
    "</div>",
    unsafe_allow_html=True,
)
