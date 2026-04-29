"""
São Paulo – Deputado Federal 2018 & 2022
Dashboard de Análise Eleitoral
"""
import os, json, warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SP – Deputado Federal 2018 & 2022",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(__file__)
AGG  = os.path.join(ROOT, "Data", "agg")
GEO  = os.path.join(ROOT, "geo_cache", "sp_municipios.geojson")

# ── Political spectrum mapping ────────────────────────────────────────────────
SPECTRUM = {
    # Esquerda
    "PT":   "Esquerda", "PSOL": "Esquerda", "PCdoB": "Esquerda",
    "PDT":  "Esquerda", "PSB":  "Esquerda", "REDE": "Esquerda",
    "UP":   "Esquerda", "PCB":  "Esquerda", "PCO":  "Esquerda",
    "PSTU": "Esquerda", "AVANTE": "Esquerda",
    # Centro-esquerda
    "MDB":  "Centro",   "PSD":  "Centro",   "PODE": "Centro",
    "SOLIDARIEDADE": "Centro", "PROS": "Centro", "PP": "Centro",
    "PL":   "Direita",  "PMN":  "Centro",
    # Centro
    "PSDB": "Centro",   "DEM":  "Centro",   "CIDADANIA": "Centro",
    "REPUBLICANOS": "Centro", "PMB": "Centro", "DC": "Centro",
    "AGIR": "Centro",   "PRTB": "Centro",
    # Direita
    "PSL":  "Direita",  "NOVO": "Direita",  "PSC":  "Direita",
    "PTC":  "Direita",  "PTB":  "Direita",  "PATRIOTA": "Direita",
    "PPL":  "Direita",  "PRP":  "Direita",  "PSB":  "Esquerda",
    "PRB":  "Direita",  "DEM":  "Direita",  "UNIAO": "Direita",
    "PEN":  "Direita",
}
SPECTRUM_COLORS = {"Esquerda": "#e63946", "Centro": "#6c757d", "Direita": "#1d3557"}

ELECTED_VALUES = {"ELEITO POR QP", "ELEITO POR MÉDIA", "ELEITO POR MEDIA"}

AGE_LABELS = ["<35", "35–44", "45–54", "55+"]


# ── Data loaders ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_resumo(year: int) -> pd.DataFrame:
    df = pd.read_csv(
        os.path.join(AGG, f"resumo_candidato_{year}.csv"),
        dtype=str, encoding="utf-8-sig",
    )
    df["ANO"] = year
    df["NR_CANDIDATO"] = df["NR_CANDIDATO"].str.strip()
    # Merge suffixes: both votes and cand files had DS_SIT_TOT_TURNO → _x/_y
    if "DS_SIT_TOT_TURNO" not in df.columns:
        src = "DS_SIT_TOT_TURNO_x" if "DS_SIT_TOT_TURNO_x" in df.columns else "DS_SIT_TOT_TURNO_y"
        df = df.rename(columns={src: "DS_SIT_TOT_TURNO"})
    # Drop the leftover duplicate suffix column if present
    for c in ["DS_SIT_TOT_TURNO_x", "DS_SIT_TOT_TURNO_y"]:
        if c in df.columns:
            df = df.drop(columns=[c])
    for col in ["QT_VOTOS_NOMINAIS", "TOTAL_DESPESAS", "TOTAL_RECEITAS", "CUSTO_POR_VOTO"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Election status
    df["ELEITO"] = df["DS_SIT_TOT_TURNO"].str.upper().isin(ELECTED_VALUES)
    df["STATUS"] = df["DS_SIT_TOT_TURNO"].apply(
        lambda x: "Eleito" if str(x).upper() in ELECTED_VALUES
        else ("Suplente" if "SUPLENTE" in str(x).upper() else "Não eleito")
    )
    # Spectrum
    df["ESPECTRO"] = df["SG_PARTIDO"].map(SPECTRUM).fillna("Centro")
    # Incumbent status
    if "DS_OCUPACAO" in df.columns:
        def _mandato(ocup):
            o = str(ocup).upper()
            if "DEPUTADO FEDERAL" in o:
                return "Reeleição"
            elif "DEPUTADO ESTADUAL" in o or "DEPUTADO DISTRITAL" in o:
                return "Dep. Estadual"
            else:
                return "Sem mandato"
        df["MANDATO_ANTERIOR"] = df["DS_OCUPACAO"].apply(_mandato)
    else:
        df["MANDATO_ANTERIOR"] = "Sem mandato"
    # Age
    if "DT_NASCIMENTO" in df.columns:
        df["DT_NASCIMENTO"] = pd.to_datetime(df["DT_NASCIMENTO"], format="%d/%m/%Y", errors="coerce")
        ref = pd.Timestamp(f"{year}-10-02")
        df["IDADE"] = ((ref - df["DT_NASCIMENTO"]).dt.days / 365.25).round(0)
        df["FAIXA_ETARIA"] = pd.cut(
            df["IDADE"], bins=[0, 35, 45, 55, 120],
            labels=AGE_LABELS, right=False,
        ).astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_votos_municipio_agregado(year: int) -> pd.DataFrame:
    """Total votes per municipality (all candidates). ~645 rows."""
    df = pd.read_csv(
        os.path.join(AGG, f"votos_municipio_agregado_{year}.csv"),
        dtype=str, encoding="utf-8-sig",
    )
    df["QT_VOTOS_NOMINAIS"] = pd.to_numeric(df["QT_VOTOS_NOMINAIS"], errors="coerce").fillna(0)
    return df


@st.cache_data(show_spinner=False)
def load_votos_muni_cand(year: int) -> pd.DataFrame:
    """Votes by candidate × municipality — all candidates with >0 votes. ~233K rows."""
    df = pd.read_csv(
        os.path.join(AGG, f"votos_muni_cand_{year}.csv"),
        dtype=str, encoding="utf-8-sig",
    )
    df["QT_VOTOS_NOMINAIS"] = pd.to_numeric(df["QT_VOTOS_NOMINAIS"], errors="coerce").fillna(0)
    df["NR_CANDIDATO"] = df["NR_CANDIDATO"].str.strip()
    return df


@st.cache_data(show_spinner=False)
def load_despesas(year: int) -> pd.DataFrame:
    df = pd.read_csv(
        os.path.join(AGG, f"despesas_por_categoria_{year}.csv"),
        dtype=str, encoding="utf-8-sig",
    )
    df["VR_TOTAL"] = pd.to_numeric(df["VR_TOTAL"], errors="coerce")
    df["NR_CANDIDATO"] = df["NR_CANDIDATO"].str.strip()
    return df


@st.cache_data(show_spinner=False)
def load_receitas(year: int) -> pd.DataFrame:
    df = pd.read_csv(
        os.path.join(AGG, f"receitas_por_fonte_{year}.csv"),
        dtype=str, encoding="utf-8-sig",
    )
    df["VR_TOTAL"] = pd.to_numeric(df["VR_TOTAL"], errors="coerce")
    df["NR_CANDIDATO"] = df["NR_CANDIDATO"].str.strip()
    return df


def _norm_city(name: str) -> str:
    """
    Robust city-name key: uppercase, strip accents, apostrophes, hyphens,
    extra spaces, and known TSE/IBGE spelling divergences (e.g. LUÍS→LUIZ).
    """
    import unicodedata
    s = str(name).upper().strip()
    # Remove accents
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    # Replace apostrophes (straight and curly) and hyphens with spaces
    s = s.replace("'", " ").replace("\u2019", " ").replace("-", " ")
    # Collapse whitespace
    s = " ".join(s.split())
    # TSE uses "LUIS" for cities where IBGE/GeoJSON uses "LUIZ" (e.g. São Luís→Luiz)
    parts = ["LUIZ" if p == "LUIS" else p for p in s.split()]
    return " ".join(parts)


@st.cache_data(show_spinner=False)
def load_municipios_geo():
    """Return (properties_df, raw_geojson_dict) — no geopandas required."""
    with open(GEO, encoding="utf-8") as f:
        geojson = json.load(f)
    rows = []
    for i, feature in enumerate(geojson["features"]):
        p = feature["properties"]
        rows.append({
            "feat_idx": i,
            "NM_MUNICIPIO": p.get("NM_MUNICIPIO", ""),
            "CD_MUNICIPIO": p.get("CD_MUNICIPIO", ""),
            "NM_UPPER": _norm_city(p.get("NM_MUNICIPIO", "")),
        })
    return pd.DataFrame(rows), geojson


# ── Helpers ───────────────────────────────────────────────────────────────────
def build_choropleth(agg_mun: pd.DataFrame, value_col: str = "QT_VOTOS_NOMINAIS"):
    """
    Join vote/metric data to GeoJSON purely with pandas+json (no geopandas).
    Returns (merged_df, annotated_geojson) ready for px.choropleth_mapbox.
    Use locations='feat_idx', featureidkey='properties.feat_idx'.
    """
    props_df, geojson = load_municipios_geo()
    agg_mun = agg_mun.copy()
    agg_mun["NM_UPPER"] = agg_mun["NM_MUNICIPIO"].apply(_norm_city)
    # Drop NM_MUNICIPIO from agg_mun to avoid _x/_y suffix collision on merge
    # (props_df already carries NM_MUNICIPIO)
    agg_mun = agg_mun.drop(columns=["NM_MUNICIPIO"], errors="ignore")
    merged = props_df.merge(agg_mun, on="NM_UPPER", how="left")
    if value_col in merged.columns:
        merged[value_col] = merged[value_col].fillna(0)
    # Stamp feat_idx into each GeoJSON feature so plotly can match rows
    # (mutation is idempotent — same index every call)
    for _, row in merged.iterrows():
        geojson["features"][int(row["feat_idx"])]["properties"]["feat_idx"] = int(row["feat_idx"])
    return merged, geojson


def fmt_brl(v):
    if pd.isna(v): return "—"
    if abs(v) >= 1_000_000: return f"R$ {v/1_000_000:.1f}M"
    if abs(v) >= 1_000:     return f"R$ {v/1_000:.0f}K"
    return f"R$ {v:,.0f}"


def remove_outliers_iqr(df: pd.DataFrame, col: str, k: float = 2.5) -> pd.DataFrame:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    return df[(df[col] >= q1 - k * iqr) & (df[col] <= q3 + k * iqr)]


def shorten(label: str, maxlen: int = 30) -> str:
    s = str(label).replace("#NULO", "Outros").replace("#NULO#", "Outros")
    return s[:maxlen] + "…" if len(s) > maxlen else s


STATUS_ORDER  = ["Eleito", "Suplente", "Não eleito"]
STATUS_COLORS = {"Eleito": "#2ecc71", "Suplente": "#f39c12", "Não eleito": "#bdc3c7"}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/Bandeira_do_estado_de_S%C3%A3o_Paulo.svg/200px-Bandeira_do_estado_de_S%C3%A3o_Paulo.svg.png", width=60)
    st.title("SP – Dep. Federal")
    st.caption("Eleições 2018 & 2022")

    year_sel = st.selectbox("Ano", [2022, 2018], index=0)
    compare_mode = st.checkbox("Comparar 2018 vs 2022", value=False)

    st.divider()
    st.subheader("Filtros")

    with st.spinner("Carregando dados…"):
        resumo22 = load_resumo(2022)
        resumo18 = load_resumo(2018)

    resumo_sel = resumo22 if year_sel == 2022 else resumo18
    all_parties = sorted(resumo_sel["SG_PARTIDO"].dropna().unique())
    parties_sel = st.multiselect("Partido", all_parties, default=[], placeholder="Todos")

    spectra = ["Esquerda", "Centro", "Direita"]
    spectra_sel = st.multiselect("Espectro", spectra, default=[], placeholder="Todos")

    genders = sorted(resumo_sel["DS_GENERO"].dropna().unique()) if "DS_GENERO" in resumo_sel.columns else []
    gender_sel = st.multiselect("Gênero", genders, default=[], placeholder="Todos")

    age_sel = st.multiselect("Faixa etária", AGE_LABELS, default=[], placeholder="Todas")

    status_sel = st.multiselect("Status", STATUS_ORDER, default=[], placeholder="Todos")

    MANDATO_ORDER = ["Reeleição", "Dep. Estadual", "Sem mandato"]
    mandato_sel = st.multiselect("Mandato anterior", MANDATO_ORDER, default=[], placeholder="Todos")

    _max_votos = int(resumo_sel["QT_VOTOS_NOMINAIS"].max()) if "QT_VOTOS_NOMINAIS" in resumo_sel.columns else 500_000
    min_votos = st.number_input(
        "Mínimo de votos", min_value=0, max_value=_max_votos,
        value=1000, step=500,
    )

    # Candidate multi-select (filtered by other selections above)
    _cand_pool = resumo_sel.copy()
    if parties_sel:  _cand_pool = _cand_pool[_cand_pool["SG_PARTIDO"].isin(parties_sel)]
    if spectra_sel:  _cand_pool = _cand_pool[_cand_pool["ESPECTRO"].isin(spectra_sel)]
    if gender_sel and "DS_GENERO" in _cand_pool.columns:
        _cand_pool = _cand_pool[_cand_pool["DS_GENERO"].isin(gender_sel)]
    if age_sel and "FAIXA_ETARIA" in _cand_pool.columns:
        _cand_pool = _cand_pool[_cand_pool["FAIXA_ETARIA"].isin(age_sel)]
    if status_sel:   _cand_pool = _cand_pool[_cand_pool["STATUS"].isin(status_sel)]
    if mandato_sel and "MANDATO_ANTERIOR" in _cand_pool.columns:
        _cand_pool = _cand_pool[_cand_pool["MANDATO_ANTERIOR"].isin(mandato_sel)]
    _cand_pool = _cand_pool[pd.to_numeric(_cand_pool["QT_VOTOS_NOMINAIS"], errors="coerce").fillna(0) >= min_votos]
    _cand_pool = _cand_pool.sort_values("QT_VOTOS_NOMINAIS", ascending=False)
    _nm_col = "NM_URNA_CANDIDATO" if "NM_URNA_CANDIDATO" in _cand_pool.columns else "NM_CANDIDATO"
    _cand_options = [
        f"{row[_nm_col]} ({row['SG_PARTIDO']})"
        for _, row in _cand_pool.iterrows()
    ]
    cands_sel = st.multiselect(
        "Candidatos", _cand_options, default=[],
        placeholder="Todos os candidatos",
        help="Deixe em branco para incluir todos. Selecione um ou mais para focar.",
    )
    # Map back to NR_CANDIDATO
    _cand_nr_sel = set()
    if cands_sel:
        _label_to_nr = {
            f"{row[_nm_col]} ({row['SG_PARTIDO']})": row["NR_CANDIDATO"]
            for _, row in _cand_pool.iterrows()
        }
        _cand_nr_sel = {_label_to_nr[lbl] for lbl in cands_sel if lbl in _label_to_nr}

    st.divider()
    st.caption("Fonte: TSE Dados Abertos")


def apply_filters(df: pd.DataFrame, skip_threshold: bool = False) -> pd.DataFrame:
    """Apply sidebar filters.  skip_threshold=True omits the min-votes cut
    so map aggregations always cover every municipality."""
    if parties_sel:  df = df[df["SG_PARTIDO"].isin(parties_sel)]
    if spectra_sel:  df = df[df["ESPECTRO"].isin(spectra_sel)]
    if gender_sel and "DS_GENERO" in df.columns:
        df = df[df["DS_GENERO"].isin(gender_sel)]
    if age_sel and "FAIXA_ETARIA" in df.columns:
        df = df[df["FAIXA_ETARIA"].isin(age_sel)]
    if status_sel:   df = df[df["STATUS"].isin(status_sel)]
    if mandato_sel and "MANDATO_ANTERIOR" in df.columns:
        df = df[df["MANDATO_ANTERIOR"].isin(mandato_sel)]
    if not skip_threshold and "QT_VOTOS_NOMINAIS" in df.columns:
        df = df[pd.to_numeric(df["QT_VOTOS_NOMINAIS"], errors="coerce").fillna(0) >= min_votos]
    if _cand_nr_sel: df = df[df["NR_CANDIDATO"].isin(_cand_nr_sel)]
    return df


# ── Descritivo ───────────────────────────────────────────────────────────────
with st.expander("ℹ️ Como usar esta ferramenta para planejamento eleitoral", expanded=False):
    st.markdown("""
Esta plataforma reúne dados das eleições de **Deputado Federal por São Paulo em 2018 e 2022** (TSE Dados Abertos)
e foi concebida para apoiar decisões estratégicas de campanha em três frentes principais:

---

#### 🗺️ Mapa de Votação — onde estão os votos?
Visualize a distribuição geográfica dos votos por município para qualquer recorte de candidatos, partidos ou espectro político.
Use para identificar **redutos eleitorais** de adversários ou aliados, detectar municípios sub-representados onde há espaço para crescimento,
e orientar a alocação de caravanas, eventos e mídia regional.

#### 💰 Custo por Voto — eficiência de campanha
Compare quanto cada candidato gastou por voto obtido e como isso varia por partido, espectro e status.
Use para **calibrar o orçamento** da campanha com base em benchmarks reais do estado,
identificar perfis de candidatos que obtiveram grande volume de votos com baixo custo (eficiência)
e dimensionar o retorno esperado de diferentes níveis de investimento.

#### 📊 Finanças — de onde vem e para onde vai o dinheiro
Explore o fluxo completo de receitas e despesas de qualquer candidato via diagrama Sankey.
Use para entender **quais fontes de financiamento** dominam em cada segmento (partido, espectro, status)
e como candidatos eleitos distribuem seus gastos entre tipos de despesa — insumo direto para a construção do plano financeiro.

#### 📋 Consolidado — retrato do campo eleitoral
Panorama completo do eleitorado de deputados federais: distribuição por status, partido, espectro, gênero, raça e faixa etária.
Use para mapear **quem compõe a bancada**, identificar sub-representações e construir argumentos de posicionamento.

#### ⚖️ 2018 vs 2022 — tendências eleitorais
Compare as duas últimas eleições para identificar partidos em ascensão ou declínio, evolução do custo de campanha,
mudanças na composição de gênero e espectro da bancada, e candidatos que se reelegeram.
Use para antecipar **dinâmicas do próximo ciclo eleitoral**.

---

**Dica:** use os filtros na barra lateral para isolar um partido, espectro, grupo demográfico ou candidatos específicos —
todos os gráficos se atualizam simultaneamente.
""")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tabs = st.tabs(["🗺️ Mapa de Votação", "💰 Custo por Voto", "📊 Finanças",
                "📋 Consolidado", "⚖️ 2018 vs 2022",
                "🏙️ Por Município", "💸 CPV por Município"])

tab_map, tab_cpv, tab_fin, tab_cons, tab_cmp, tab_mun, tab_cpv_mun = tabs


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 – MAPA DE VOTAÇÃO
# ════════════════════════════════════════════════════════════════════════════
with tab_map:
    st.header(f"Mapa de Votação – {year_sel}")
    resumo = apply_filters(resumo_sel, skip_threshold=True)

    color_mode = st.radio(
        "Cor representa",
        ["Votos obtidos no município", "% dos votos dep. federal no município"],
        horizontal=True,
    )

    # Aggregate filtered candidates → per municipality
    votos_all = load_votos_muni_cand(year_sel)
    votos_filt = votos_all[votos_all["NR_CANDIDATO"].isin(resumo["NR_CANDIDATO"])]
    agg_mun = votos_filt.groupby("NM_MUNICIPIO")["QT_VOTOS_NOMINAIS"].sum().reset_index()

    # Filter summary caption
    n_cands   = resumo["NR_CANDIDATO"].nunique()
    n_parties = resumo["SG_PARTIDO"].nunique()
    total_v   = int(resumo["QT_VOTOS_NOMINAIS"].sum()) if "QT_VOTOS_NOMINAIS" in resumo.columns else 0
    st.caption(f"Recorte: **{n_cands} candidatos** · **{n_parties} partidos** · **{total_v:,} votos totais**")

    if color_mode == "% dos votos dep. federal no município":
        total_mun = (
            load_votos_municipio_agregado(year_sel)
            [["NM_MUNICIPIO", "QT_VOTOS_NOMINAIS"]]
            .rename(columns={"QT_VOTOS_NOMINAIS": "TOTAL_MUN"})
        )
        agg_mun = agg_mun.merge(total_mun, on="NM_MUNICIPIO", how="left")
        agg_mun["PCT"] = (
            agg_mun["QT_VOTOS_NOMINAIS"] / agg_mun["TOTAL_MUN"].replace(0, np.nan) * 100
        ).round(2).fillna(0)
        color_col   = "PCT"
        color_label = "% dos votos"
        color_scale = "Blues"
        map_title   = f"% dos votos dep. federal obtidos no município – {year_sel}"
    else:
        color_col   = "QT_VOTOS_NOMINAIS"
        color_label = "Votos"
        color_scale = "YlOrRd"
        map_title   = f"Votos obtidos por município – {year_sel}"

    merged, geojson = build_choropleth(agg_mun, value_col=color_col)

    if merged[color_col].sum() == 0:
        st.info("Nenhum voto para os filtros selecionados.")
    else:
        hover_extra = {"PCT": ":.1f"} if color_col == "PCT" else {}
        fig = px.choropleth_mapbox(
            merged,
            geojson=geojson,
            locations="feat_idx",
            featureidkey="properties.feat_idx",
            color=color_col,
            color_continuous_scale=color_scale,
            mapbox_style="carto-positron",
            zoom=6, center={"lat": -22.5, "lon": -48.5},
            opacity=0.75,
            hover_data={"NM_MUNICIPIO": True, "QT_VOTOS_NOMINAIS": True, **hover_extra},
            labels={color_col: color_label, "QT_VOTOS_NOMINAIS": "Votos"},
            title=map_title,
        )
        fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0}, height=600)
        st.plotly_chart(fig, use_container_width=True)

        all_mun_disp = agg_mun.sort_values(color_col, ascending=False).copy()
        all_mun_disp = all_mun_disp.rename(columns={"NM_MUNICIPIO": "Município", "QT_VOTOS_NOMINAIS": "Votos"})
        if color_col == "PCT":
            all_mun_disp["% no mun."] = all_mun_disp["PCT"].map("{:.1f}%".format)
        st.subheader(f"Municípios ({len(all_mun_disp)})")
        st.dataframe(all_mun_disp.reset_index(drop=True), hide_index=True, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 – CUSTO POR VOTO
# ════════════════════════════════════════════════════════════════════════════
with tab_cpv:
    st.header(f"Custo por Voto – {year_sel}")
    resumo = apply_filters(resumo_sel).dropna(subset=["QT_VOTOS_NOMINAIS", "TOTAL_DESPESAS"])

    scope = st.radio(
        "Escopo", ["Apenas eleitos", "Eleitos e suplentes", "Todos"],
        horizontal=True, index=2, key="cpv_scope",
    )
    if scope == "Apenas eleitos":
        resumo = resumo[resumo["ELEITO"]]
    elif scope == "Eleitos e suplentes":
        resumo = resumo[resumo["STATUS"].isin(["Eleito", "Suplente"])]

    if resumo.empty:
        st.info("Sem dados para os filtros selecionados.")
    else:
        # ── Scatter ──────────────────────────────────────────────────────────
        st.subheader("Votos vs. Custo por Voto")
        scatter_df = remove_outliers_iqr(resumo.dropna(subset=["CUSTO_POR_VOTO"]), "CUSTO_POR_VOTO")
        scatter_df["NM_LABEL"] = scatter_df["NM_URNA_CANDIDATO"] if "NM_URNA_CANDIDATO" in scatter_df.columns else scatter_df["NM_CANDIDATO"]
        fig = px.scatter(
            scatter_df, x="QT_VOTOS_NOMINAIS", y="CUSTO_POR_VOTO",
            color="ESPECTRO", symbol="STATUS",
            color_discrete_map=SPECTRUM_COLORS,
            size="QT_VOTOS_NOMINAIS", size_max=30,
            hover_name="NM_LABEL",
            hover_data={"SG_PARTIDO": True, "TOTAL_DESPESAS": ":,.0f",
                        "QT_VOTOS_NOMINAIS": ":,", "ESPECTRO": False},
            labels={
                "QT_VOTOS_NOMINAIS": "Votos nominais",
                "TOTAL_DESPESAS": "Gastos (R$)",
                "CUSTO_POR_VOTO": "Custo/voto (R$)",
            },
        )
        fig.update_layout(height=500, legend_title="Espectro / Status")
        st.plotly_chart(fig, use_container_width=True)

        # ── Box plots ─────────────────────────────────────────────────────────
        col1, col2 = st.columns(2)
        for col_idx, (metric, label, unit) in enumerate([
            ("CUSTO_POR_VOTO", "Custo por Voto", "R$"),
            ("TOTAL_DESPESAS", "Gastos Totais",  "R$"),
        ]):
            bx = resumo.dropna(subset=[metric])
            bx = remove_outliers_iqr(bx, metric)
            if bx.empty:
                continue
            target_col = col1 if col_idx == 0 else col2
            with target_col:
                for grp_col, grp_title in [("STATUS", "por Status"), ("ESPECTRO", "por Espectro")]:
                    fig = px.box(
                        bx, x=grp_col, y=metric,
                        color=grp_col,
                        color_discrete_map={**STATUS_COLORS, **SPECTRUM_COLORS},
                        points="outliers",
                        category_orders={
                            "STATUS": STATUS_ORDER,
                            "ESPECTRO": ["Esquerda", "Centro", "Direita"],
                        },
                        title=f"{label} {grp_title} – {year_sel}",
                        labels={metric: f"{label} ({unit})", grp_col: ""},
                    )
                    fig.update_layout(height=380, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

        # ── Bar: avg cost/vote per party ─────────────────────────────────────
        st.subheader("Custo médio por voto — por partido")
        bar_scope = st.radio(
            "Escopo (gráfico de barras)",
            ["Apenas eleitos", "Eleitos e suplentes", "Todos"],
            horizontal=True, index=0, key="bar_scope_cpv",
        )
        if bar_scope == "Apenas eleitos":
            el = resumo[resumo["ELEITO"]]
        elif bar_scope == "Eleitos e suplentes":
            el = resumo[resumo["STATUS"].isin(["Eleito", "Suplente"])]
        else:
            el = resumo.copy()
        el = el.dropna(subset=["CUSTO_POR_VOTO"])
        el = remove_outliers_iqr(el, "CUSTO_POR_VOTO")
        bar_mode = st.radio("Ordenar por", ["Custo/voto", "Partido"], horizontal=True, key="bar_mode_cpv")
        avg_party = el.groupby("SG_PARTIDO")["CUSTO_POR_VOTO"].mean().reset_index()
        avg_party.columns = ["Partido", "Custo/voto (R$)"]
        if bar_mode == "Custo/voto":
            avg_party = avg_party.sort_values("Custo/voto (R$)")
        else:
            avg_party = avg_party.sort_values("Partido")
        avg_party["Espectro"] = avg_party["Partido"].map(SPECTRUM).fillna("Centro")
        scope_label = {"Apenas eleitos": "eleitos", "Eleitos e suplentes": "eleitos + suplentes", "Todos": "todos"}[bar_scope]
        fig = px.bar(
            avg_party, x="Partido", y="Custo/voto (R$)",
            color="Espectro", color_discrete_map=SPECTRUM_COLORS,
            title=f"Custo médio por voto por partido ({scope_label}) – {year_sel}",
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 – FINANÇAS (SANKEY)
# ════════════════════════════════════════════════════════════════════════════
with tab_fin:
    st.header(f"Finanças de Campanha – {year_sel}")

    resumo = apply_filters(resumo_sel)
    cand_opts = {
        f"{r.get('NM_URNA_CANDIDATO') or r['NM_CANDIDATO']} ({r['SG_PARTIDO']}) – {r['STATUS']}": r["NR_CANDIDATO"]
        for _, r in resumo.sort_values("QT_VOTOS_NOMINAIS", ascending=False).head(200).iterrows()
    }
    if not cand_opts:
        st.info("Nenhum candidato disponível para os filtros selecionados.")
        st.stop()

    cand_label = st.selectbox("Candidato", list(cand_opts.keys()), key="fin_cand")
    nr_sel = cand_opts[cand_label]

    rec = load_receitas(year_sel)
    desp = load_despesas(year_sel)
    rec_cand  = rec[rec["NR_CANDIDATO"]  == nr_sel]
    desp_cand = desp[desp["NR_CANDIDATO"] == nr_sel]

    if rec_cand.empty and desp_cand.empty:
        st.info("Sem dados financeiros para este candidato.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Receitas",   fmt_brl(rec_cand["VR_TOTAL"].sum()))
        with col2:
            st.metric("Total Despesas",   fmt_brl(desp_cand["VR_TOTAL"].sum()))

        # Sankey: Fontes → CANDIDATO → Categorias de despesa
        tot_rec  = rec_cand["VR_TOTAL"].sum() or 1
        tot_desp = desp_cand["VR_TOTAL"].sum() or 1
        MIN_SHARE = 0.02  # hide sources/categories < 2%

        rec_filt  = rec_cand[rec_cand["VR_TOTAL"]  / tot_rec  >= MIN_SHARE]
        desp_filt = desp_cand[desp_cand["VR_TOTAL"] / tot_desp >= MIN_SHARE]

        rec_agg  = rec_filt.groupby("DS_ORIGEM_RECEITA")["VR_TOTAL"].sum()
        desp_agg = desp_filt.groupby("DS_ORIGEM_DESPESA")["VR_TOTAL"].sum()

        sources  = [shorten(s) for s in rec_agg.index]
        targets  = [shorten(s) for s in desp_agg.index]
        mid_node = "Campanha"

        nodes = sources + [mid_node] + targets
        node_idx = {n: i for i, n in enumerate(nodes)}

        link_src  = [node_idx[s] for s in sources]
        link_tgt  = [node_idx[mid_node]] * len(sources)
        link_vals = rec_agg.values.tolist()

        link_src  += [node_idx[mid_node]] * len(targets)
        link_tgt  += [node_idx[t] for t in targets]
        link_vals += desp_agg.values.tolist()

        node_colors = (
            ["#2980b9"] * len(sources)   # receitas – blue
            + ["#2c3e50"]                # campanha – dark
            + ["#c0392b"] * len(targets) # despesas – red
        )
        fig = go.Figure(go.Sankey(
            arrangement="snap",
            node=dict(
                label=nodes,
                color=node_colors,
                line=dict(color="white", width=1),
                pad=25,
                thickness=18,
            ),
            link=dict(
                source=link_src, target=link_tgt, value=link_vals,
                color="rgba(180,180,180,0.25)",
            ),
            textfont=dict(size=12, color="black", family="Arial, sans-serif"),
        ))
        fig.update_layout(
            title_text=f"Fluxo Financeiro – {cand_label.split('(')[0].strip()}",
            height=560,
            paper_bgcolor="white",
            font=dict(size=12, color="black", family="Arial, sans-serif"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Bar charts: revenue breakdown and expense breakdown
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Receitas por origem")
            rec_bar = rec_cand.groupby("DS_ORIGEM_RECEITA")["VR_TOTAL"].sum().sort_values(ascending=True)
            rec_bar.index = [shorten(s, 40) for s in rec_bar.index]
            fig = px.bar(rec_bar, orientation="h",
                         labels={"value": "R$", "index": ""},
                         color_discrete_sequence=["#3498db"])
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Despesas por categoria")
            desp_bar = desp_cand.groupby("DS_ORIGEM_DESPESA")["VR_TOTAL"].sum().sort_values(ascending=True)
            desp_bar.index = [shorten(s, 40) for s in desp_bar.index]
            fig = px.bar(desp_bar, orientation="h",
                         labels={"value": "R$", "index": ""},
                         color_discrete_sequence=["#e74c3c"])
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 – CONSOLIDADO
# ════════════════════════════════════════════════════════════════════════════
with tab_cons:
    st.header(f"Consolidado – {year_sel}")
    resumo = apply_filters(resumo_sel)

    if resumo.empty:
        st.info("Nenhum candidato para os filtros selecionados.")
    else:
        eleitos  = resumo[resumo["ELEITO"]]
        n_total  = len(resumo)
        n_eleito = len(eleitos)

        # KPIs
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Candidatos", f"{n_total:,}")
        k2.metric("Eleitos", f"{n_eleito}")
        k3.metric("Votos totais", f"{resumo['QT_VOTOS_NOMINAIS'].sum():,.0f}")
        k4.metric("Gasto médio (eleitos)", fmt_brl(eleitos["TOTAL_DESPESAS"].mean()))
        k5.metric("Custo/voto médio (eleitos)", f"R$ {eleitos['CUSTO_POR_VOTO'].median():.2f}")

        st.divider()

        # Row 1: Status + Spectrum
        c1, c2 = st.columns(2)
        with c1:
            by_status = resumo["STATUS"].value_counts().reindex(STATUS_ORDER).dropna()
            fig = px.bar(by_status, color=by_status.index,
                         color_discrete_map=STATUS_COLORS,
                         title="Candidatos por status",
                         labels={"value": "Candidatos", "index": "Status"})
            fig.update_layout(showlegend=False, height=320)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            by_spec = resumo.groupby("ESPECTRO")["ELEITO"].agg(["sum", "count"]).reset_index()
            by_spec.columns = ["Espectro", "Eleitos", "Total"]
            by_spec["Taxa (%)"] = (by_spec["Eleitos"] / by_spec["Total"] * 100).round(1)
            fig = px.bar(by_spec, x="Espectro", y="Eleitos",
                         color="Espectro", color_discrete_map=SPECTRUM_COLORS,
                         text="Taxa (%)", title="Eleitos por espectro",
                         labels={"Eleitos": "Eleitos"})
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(showlegend=False, height=320)
            st.plotly_chart(fig, use_container_width=True)

        # Row 2: Party breakdown (elected)
        st.subheader("Eleitos por partido")
        by_party = (
            eleitos.groupby("SG_PARTIDO")
            .agg(Eleitos=("NR_CANDIDATO", "count"),
                 Votos=("QT_VOTOS_NOMINAIS", "sum"),
                 Gasto_Medio=("TOTAL_DESPESAS", "mean"),
                 CPV_Medio=("CUSTO_POR_VOTO", "mean"))
            .reset_index().sort_values("Eleitos", ascending=False)
        )
        by_party["Espectro"] = by_party["SG_PARTIDO"].map(SPECTRUM).fillna("Centro")
        fig = px.bar(
            by_party.head(20), x="SG_PARTIDO", y="Eleitos",
            color="Espectro", color_discrete_map=SPECTRUM_COLORS,
            text="Eleitos", hover_data={"Votos": ":,", "Gasto_Medio": ":,.0f"},
            title=f"Eleitos por partido – {year_sel}",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(height=380, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        # Row 3: Gender & Race (elected)
        c1, c2 = st.columns(2)
        with c1:
            if "DS_GENERO" in eleitos.columns:
                gen = eleitos["DS_GENERO"].value_counts()
                fig = px.pie(values=gen.values, names=gen.index,
                             title="Eleitos por gênero",
                             color_discrete_sequence=px.colors.qualitative.Set2)
                fig.update_layout(height=320)
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            if "DS_COR_RACA" in eleitos.columns:
                raca = eleitos["DS_COR_RACA"].value_counts().head(6)
                fig = px.pie(values=raca.values, names=raca.index,
                             title="Eleitos por cor/raça",
                             color_discrete_sequence=px.colors.qualitative.Pastel)
                fig.update_layout(height=320)
                st.plotly_chart(fig, use_container_width=True)

        # Row 4: Age distribution
        if "FAIXA_ETARIA" in resumo.columns:
            st.subheader("Distribuição etária")
            age_comp = (
                resumo.groupby(["FAIXA_ETARIA", "STATUS"])
                .size().reset_index(name="N")
            )
            fig = px.bar(
                age_comp, x="FAIXA_ETARIA", y="N",
                color="STATUS", barmode="group",
                color_discrete_map=STATUS_COLORS,
                category_orders={"FAIXA_ETARIA": AGE_LABELS, "STATUS": STATUS_ORDER},
                title="Candidatos por faixa etária e status",
                labels={"N": "Candidatos", "FAIXA_ETARIA": "Faixa etária"},
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        # Row 5: Custo/voto by spectrum (elected)
        st.subheader("Custo médio por voto — por espectro (eleitos)")
        cpv_spec = (
            eleitos.dropna(subset=["CUSTO_POR_VOTO"])
            .pipe(lambda d: remove_outliers_iqr(d, "CUSTO_POR_VOTO"))
            .groupby("ESPECTRO")["CUSTO_POR_VOTO"].mean().reset_index()
        )
        cpv_spec.columns = ["Espectro", "Custo/voto (R$)"]
        fig = px.bar(
            cpv_spec, x="Espectro", y="Custo/voto (R$)",
            color="Espectro", color_discrete_map=SPECTRUM_COLORS,
            text_auto=".2f", title=f"Custo médio por voto por espectro (eleitos) – {year_sel}",
        )
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

        # Row 6: Aggregate map
        st.subheader("Mapa agregado de votação")
        agg_mun_map = load_votos_municipio_agregado(year_sel)[["NM_MUNICIPIO", "QT_VOTOS_NOMINAIS"]]
        merged_map, geojson_map = build_choropleth(agg_mun_map)
        if merged_map["QT_VOTOS_NOMINAIS"].sum() > 0:
            fig = px.choropleth_mapbox(
                merged_map, geojson=geojson_map,
                locations="feat_idx", featureidkey="properties.feat_idx",
                color="QT_VOTOS_NOMINAIS",
                color_continuous_scale="Blues",
                mapbox_style="carto-positron",
                zoom=6, center={"lat": -22.5, "lon": -48.5},
                opacity=0.75,
                hover_data={"NM_MUNICIPIO": True, "QT_VOTOS_NOMINAIS": True},
                labels={"QT_VOTOS_NOMINAIS": "Votos"},
                title=f"Distribuição territorial de votos – {year_sel}",
            )
            fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0}, height=520)
            st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 – 2018 vs 2022
# ════════════════════════════════════════════════════════════════════════════
with tab_cmp:
    st.header("Comparativo 2018 vs 2022")

    r18 = apply_filters(resumo18)
    r22 = apply_filters(resumo22)
    combined = pd.concat([r18, r22], ignore_index=True)
    combined["ANO"] = combined["ANO"].astype(str)

    if combined.empty:
        st.info("Sem dados para os filtros selecionados.")
    else:
        # KPIs side by side
        k1, k2, k3, k4 = st.columns(4)
        for yr, df_yr, col in [(2018, r18, k1), (2022, r22, k2)]:
            el = df_yr[df_yr["ELEITO"]]
            col.metric(f"Eleitos {yr}", len(el))
        k3.metric("Partidos com assento 2018", r18[r18["ELEITO"]]["SG_PARTIDO"].nunique())
        k4.metric("Partidos com assento 2022", r22[r22["ELEITO"]]["SG_PARTIDO"].nunique())

        st.divider()

        # Eleitos por partido – side by side
        st.subheader("Eleitos por partido")
        ep = (
            combined[combined["ELEITO"]]
            .groupby(["ANO", "SG_PARTIDO"]).size().reset_index(name="Eleitos")
        )
        ep["Espectro"] = ep["SG_PARTIDO"].map(SPECTRUM).fillna("Centro")
        fig = px.bar(
            ep, x="SG_PARTIDO", y="Eleitos", color="ANO", barmode="group",
            facet_col="Espectro",
            title="Eleitos por partido – 2018 vs 2022",
            labels={"SG_PARTIDO": "Partido"},
            color_discrete_map={"2018": "#636EFA", "2022": "#EF553B"},
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

        # Cost per vote evolution
        st.subheader("Custo por voto – distribuição")
        cpv_cmp = (
            combined[combined["ELEITO"]].dropna(subset=["CUSTO_POR_VOTO"])
            .pipe(lambda d: remove_outliers_iqr(d, "CUSTO_POR_VOTO"))
        )
        fig = px.box(
            cpv_cmp, x="ANO", y="CUSTO_POR_VOTO", color="ANO",
            color_discrete_map={"2018": "#636EFA", "2022": "#EF553B"},
            points="outliers",
            title="Distribuição de custo por voto – eleitos",
            labels={"CUSTO_POR_VOTO": "R$/voto", "ANO": "Ano"},
        )
        fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Total spending evolution
        c1, c2 = st.columns(2)
        with c1:
            spend = (
                combined[combined["ELEITO"]].dropna(subset=["TOTAL_DESPESAS"])
                .groupby("ANO")["TOTAL_DESPESAS"]
                .agg(["mean", "median", "sum"]).reset_index()
            )
            spend.columns = ["Ano", "Média", "Mediana", "Total"]
            fig = px.bar(
                spend.melt(id_vars="Ano", value_vars=["Média", "Mediana"]),
                x="Ano", y="value", color="variable", barmode="group",
                title="Gasto de campanha – eleitos",
                labels={"value": "R$", "variable": ""},
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            if "DS_GENERO" in combined.columns:
                gen_yr = (
                    combined[combined["ELEITO"]]
                    .groupby(["ANO", "DS_GENERO"]).size().reset_index(name="N")
                )
                fig = px.bar(
                    gen_yr, x="ANO", y="N", color="DS_GENERO", barmode="stack",
                    title="Eleitos por gênero",
                    labels={"N": "Eleitos", "DS_GENERO": "Gênero"},
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

        # Espectro evolution
        spec_yr = (
            combined[combined["ELEITO"]]
            .groupby(["ANO", "ESPECTRO"]).size().reset_index(name="Eleitos")
        )
        fig = px.bar(
            spec_yr, x="ESPECTRO", y="Eleitos", color="ANO", barmode="group",
            color_discrete_map={"2018": "#636EFA", "2022": "#EF553B"},
            category_orders={"ESPECTRO": ["Esquerda", "Centro", "Direita"]},
            title="Eleitos por espectro – 2018 vs 2022",
        )
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

        # Candidates elected in both years
        st.subheader("Reeleitos (eleitos em 2018 e 2022)")
        el18 = set(r18[r18["ELEITO"]]["NM_CANDIDATO"].str.upper().str.strip())
        el22 = r22[r22["ELEITO"]].copy()
        el22["NM_UPPER"] = el22["NM_CANDIDATO"].str.upper().str.strip()
        reeleitos = el22[el22["NM_UPPER"].isin(el18)][
            ["NM_CANDIDATO", "SG_PARTIDO", "QT_VOTOS_NOMINAIS", "TOTAL_DESPESAS", "CUSTO_POR_VOTO"]
        ].sort_values("QT_VOTOS_NOMINAIS", ascending=False).reset_index(drop=True)
        reeleitos.columns = ["Candidato", "Partido", "Votos 2022", "Gastos 2022 (R$)", "R$/voto 2022"]
        if reeleitos.empty:
            st.info("Nenhum reeleito com os filtros aplicados.")
        else:
            st.dataframe(
                reeleitos.style.format({
                    "Votos 2022": "{:,.0f}",
                    "Gastos 2022 (R$)": "R$ {:,.0f}",
                    "R$/voto 2022": "R$ {:.2f}",
                }, na_rep="—"),
                use_container_width=True, hide_index=True,
            )


# ════════════════════════════════════════════════════════════════════════════
# TAB 6 – POR MUNICÍPIO
# ════════════════════════════════════════════════════════════════════════════
with tab_mun:
    st.header(f"Resultado por Município – {year_sel}")

    _votos_mun = load_votos_muni_cand(year_sel)
    _cidades = sorted(_votos_mun["NM_MUNICIPIO"].dropna().unique())
    cidade_sel = st.selectbox("Município", _cidades, index=_cidades.index("SAO PAULO") if "SAO PAULO" in _cidades else 0)

    resumo_mun = apply_filters(resumo_sel)
    _nm_col_m = "NM_URNA_CANDIDATO" if "NM_URNA_CANDIDATO" in resumo_mun.columns else "NM_CANDIDATO"

    # Votes in the selected city for filtered candidates
    _city_votes = (
        _votos_mun[
            (_votos_mun["NM_MUNICIPIO"] == cidade_sel) &
            (_votos_mun["NR_CANDIDATO"].isin(resumo_mun["NR_CANDIDATO"]))
        ]
        .groupby("NR_CANDIDATO")["QT_VOTOS_NOMINAIS"].sum()
        .reset_index()
    )

    # Join with resumo for candidate metadata
    _city_df = _city_votes.merge(
        resumo_mun[["NR_CANDIDATO", _nm_col_m, "SG_PARTIDO", "ESPECTRO",
                    "STATUS", "MANDATO_ANTERIOR", "QT_VOTOS_NOMINAIS",
                    "CUSTO_POR_VOTO", "TOTAL_DESPESAS"]].rename(
            columns={"QT_VOTOS_NOMINAIS": "VOTOS_TOTAL_SP"}
        ),
        on="NR_CANDIDATO", how="left",
    ).sort_values("QT_VOTOS_NOMINAIS", ascending=False)

    _city_total = _city_df["QT_VOTOS_NOMINAIS"].sum()
    _city_df["PCT_CIDADE"] = (_city_df["QT_VOTOS_NOMINAIS"] / _city_total * 100).round(2) if _city_total > 0 else 0.0

    if _city_df.empty:
        st.info("Nenhum candidato com votos neste município para os filtros selecionados.")
    else:
        # KPIs
        k1, k2, k3 = st.columns(3)
        k1.metric("Candidatos com votos", len(_city_df))
        k2.metric("Total de votos no município", f"{int(_city_total):,}")
        k3.metric("Eleitos com votos aqui", int((_city_df["STATUS"] == "Eleito").sum()))

        st.divider()

        # Bar chart – top candidates by votes in city
        _top_n = st.slider("Mostrar top N candidatos", 10, min(100, len(_city_df)), 20, key="mun_topn")
        _bar_df = _city_df.head(_top_n).copy()
        _bar_df["LABEL"] = _bar_df[_nm_col_m].str.title() + " (" + _bar_df["SG_PARTIDO"] + ")"
        fig = px.bar(
            _bar_df.sort_values("QT_VOTOS_NOMINAIS"),
            x="QT_VOTOS_NOMINAIS", y="LABEL",
            color="STATUS",
            color_discrete_map=STATUS_COLORS,
            orientation="h",
            text="QT_VOTOS_NOMINAIS",
            hover_data={"ESPECTRO": True, "MANDATO_ANTERIOR": True,
                        "PCT_CIDADE": ":.2f", "VOTOS_TOTAL_SP": ":,"},
            labels={
                "QT_VOTOS_NOMINAIS": "Votos no município",
                "LABEL": "",
                "PCT_CIDADE": "% dos votos na cidade",
                "VOTOS_TOTAL_SP": "Votos totais SP",
                "MANDATO_ANTERIOR": "Mandato anterior",
            },
            title=f"Votos por candidato – {cidade_sel.title()} ({year_sel})",
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(height=max(400, _top_n * 28), showlegend=True, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        # Full table
        st.subheader("Tabela completa")
        _disp = _city_df[[_nm_col_m, "SG_PARTIDO", "ESPECTRO", "STATUS",
                           "MANDATO_ANTERIOR", "QT_VOTOS_NOMINAIS",
                           "PCT_CIDADE", "VOTOS_TOTAL_SP",
                           "CUSTO_POR_VOTO", "TOTAL_DESPESAS"]].copy()
        _disp.columns = ["Candidato", "Partido", "Espectro", "Status",
                         "Mandato anterior", "Votos no mun.", "% da cidade",
                         "Votos totais SP", "R$/voto", "Gastos (R$)"]
        st.dataframe(
            _disp.style.format({
                "Votos no mun.":   "{:,.0f}",
                "% da cidade":     "{:.2f}%",
                "Votos totais SP": "{:,.0f}",
                "R$/voto":         lambda v: f"R$ {v:,.2f}" if pd.notna(v) else "—",
                "Gastos (R$)":     lambda v: f"R$ {v:,.0f}" if pd.notna(v) else "—",
            }),
            use_container_width=True, hide_index=True,
        )


# ════════════════════════════════════════════════════════════════════════════
# TAB 7 – CPV POR MUNICÍPIO (mapa)
# ════════════════════════════════════════════════════════════════════════════
with tab_cpv_mun:
    st.header(f"Custo médio ponderado por voto — por município – {year_sel}")
    st.caption(
        "Para cada município, calcula a média do custo por voto dos candidatos "
        "ponderada pelo número de votos que cada um recebeu naquele município. "
        "Reflete quanto custou, em média, cada voto depositado na cidade."
    )

    # skip_threshold=True so every municipality appears even if its local
    # candidates are below the sidebar vote-minimum
    _resumo_all_filt  = apply_filters(resumo_sel, skip_threshold=True)
    resumo_cpv_mun    = _resumo_all_filt.dropna(subset=["CUSTO_POR_VOTO"]) # subset with CPV data

    if _resumo_all_filt.empty:
        st.info("Sem dados para os filtros selecionados.")
    else:
        _votos_all_mun = load_votos_muni_cand(year_sel)

        # Total votes per city from ALL filtered candidates (for hover)
        _tot_city = (
            _votos_all_mun[_votos_all_mun["NR_CANDIDATO"].isin(_resumo_all_filt["NR_CANDIDATO"])]
            .groupby("NM_MUNICIPIO")["QT_VOTOS_NOMINAIS"].sum().reset_index()
            .rename(columns={"QT_VOTOS_NOMINAIS": "VOTOS_CIDADE"})
        )

        # Weighted-average CPV only for candidates who have CPV data
        _votos_cpv = _votos_all_mun[
            _votos_all_mun["NR_CANDIDATO"].isin(resumo_cpv_mun["NR_CANDIDATO"])
        ].merge(
            resumo_cpv_mun[["NR_CANDIDATO", "CUSTO_POR_VOTO"]],
            on="NR_CANDIDATO", how="inner",
        )
        _votos_cpv = _votos_cpv[_votos_cpv["QT_VOTOS_NOMINAIS"] > 0]

        # Vote-weighted average CPV per municipality
        def _wavg_cpv(g):
            total = g["QT_VOTOS_NOMINAIS"].sum()
            if total == 0:
                return np.nan
            return np.average(g["CUSTO_POR_VOTO"], weights=g["QT_VOTOS_NOMINAIS"])

        _cpv_mun = (
            _votos_cpv.groupby("NM_MUNICIPIO")
            .apply(_wavg_cpv, include_groups=False)
            .reset_index()
        )
        _cpv_mun.columns = ["NM_MUNICIPIO", "CPV_MEDIO"]
        _cpv_mun["CPV_MEDIO"] = _cpv_mun["CPV_MEDIO"].round(2)
        _cpv_mun = _cpv_mun.merge(_tot_city, on="NM_MUNICIPIO", how="right")  # keep all cities

        n_with_cpv   = _cpv_mun["CPV_MEDIO"].notna().sum()
        n_no_cpv     = _cpv_mun["CPV_MEDIO"].isna().sum()
        st.caption(
            f"**{n_with_cpv}** municípios com dados de CPV | "
            f"**{n_no_cpv}** sem dados financeiros disponíveis (exibidos em cinza)"
        )

        # Remove extreme outliers for colour scale legibility (only from cities with data)
        _cpv_clean = remove_outliers_iqr(_cpv_mun.dropna(subset=["CPV_MEDIO"]), "CPV_MEDIO", k=3)
        _vmax = _cpv_clean["CPV_MEDIO"].quantile(0.95) if not _cpv_clean.empty else 100

        merged_cpv, geojson_cpv = build_choropleth(_cpv_mun, value_col="CPV_MEDIO")

        fig = px.choropleth_mapbox(
            merged_cpv,
            geojson=geojson_cpv,
            locations="feat_idx",
            featureidkey="properties.feat_idx",
            color="CPV_MEDIO",
            color_continuous_scale="RdYlGn_r",
            range_color=[0, _vmax],
            mapbox_style="carto-positron",
            zoom=6, center={"lat": -22.5, "lon": -48.5},
            opacity=0.8,
            hover_data={"NM_MUNICIPIO": True, "CPV_MEDIO": ":.2f", "VOTOS_CIDADE": ":,"},
            labels={"CPV_MEDIO": "R$/voto médio", "VOTOS_CIDADE": "Votos contados",
                    "NM_MUNICIPIO": "Município"},
            title=f"Custo médio ponderado por voto por município – {year_sel}",
        )
        fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0}, height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Top / bottom table
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🔴 Municípios mais caros (top 15)")
            _top = _cpv_mun.dropna(subset=["CPV_MEDIO"]).sort_values("CPV_MEDIO", ascending=False).head(15)
            st.dataframe(
                _top[["NM_MUNICIPIO", "CPV_MEDIO", "VOTOS_CIDADE"]].rename(
                    columns={"NM_MUNICIPIO": "Município", "CPV_MEDIO": "R$/voto médio",
                             "VOTOS_CIDADE": "Votos"}
                ).style.format({"R$/voto médio": "R$ {:.2f}", "Votos": "{:,.0f}"}),
                use_container_width=True, hide_index=True,
            )
        with c2:
            st.subheader("🟢 Municípios mais eficientes (top 15)")
            _bot = _cpv_mun.dropna(subset=["CPV_MEDIO"]).sort_values("CPV_MEDIO").head(15)
            st.dataframe(
                _bot[["NM_MUNICIPIO", "CPV_MEDIO", "VOTOS_CIDADE"]].rename(
                    columns={"NM_MUNICIPIO": "Município", "CPV_MEDIO": "R$/voto médio",
                             "VOTOS_CIDADE": "Votos"}
                ).style.format({"R$/voto médio": "R$ {:.2f}", "Votos": "{:,.0f}"}),
                use_container_width=True, hide_index=True,
            )
