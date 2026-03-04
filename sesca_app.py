#!/usr/bin/env python3
"""
=============================================================================
  SESCA Web App - Predicao de Espectros de Dicroismo Circular (CD)
=============================================================================
Interface Streamlit para o pipeline SESCA.

Executar com:
  streamlit run sesca_app.py
"""

import tempfile
import io
from pathlib import Path

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

from sesca_core import (
    setup_sesca, check_sesca, fetch_pdb, clean_pdb,
    run_sesca, save_combined_csv,
    DEFAULT_BASIS, BASIS_OPTIONS, SESCA_DIR,
)

# =============================================================================
#  CONFIGURACAO DA PAGINA
# =============================================================================

st.set_page_config(
    page_title="SESCA | CD Spectrum Predictor",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
#  CSS CUSTOMIZADO
# =============================================================================

st.markdown("""
<style>
    /* Header principal */
    .main-header {
        background: #fafafa;
        padding: 1.8rem 2rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #e5e7eb;
    }
    .main-header h1 {
        color: #111827;
        font-size: 1.7rem;
        font-weight: 600;
        margin: 0 0 0.25rem 0;
        letter-spacing: -0.3px;
    }
    .main-header p {
        color: #6b7280;
        font-size: 0.95rem;
        margin: 0;
    }

    /* Cards de metricas */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-card .label {
        color: #374151;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    .metric-card .value {
        color: #111827;
        font-size: 1.5rem;
        font-weight: 700;
        font-family: 'SF Mono', 'Fira Code', monospace;
    }
    .metric-card .unit {
        color: #9ca3af;
        font-size: 0.8rem;
        margin-top: 0.15rem;
    }

    /* Status badge */
    .status-ok {
        color: #059669;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .status-err {
        color: #dc2626;
        font-size: 0.85rem;
        font-weight: 500;
    }

    /* Esconder menu e footer padrao */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 8px 20px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
#  SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("## Configuracoes")
    st.markdown("---")

    # Setup automatico do SESCA
    if "sesca_setup_done" not in st.session_state:
        with st.spinner("Configurando SESCA..."):
            st.session_state["sesca_setup_done"] = setup_sesca(
                log=lambda msg: None
            )

    sesca_ready = check_sesca() and st.session_state.get("sesca_setup_done", False)
    if sesca_ready:
        st.markdown('<span class="status-ok">SESCA pronto</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-err">SESCA nao disponivel</span>', unsafe_allow_html=True)
        if st.button("Reconfigurar SESCA", use_container_width=True):
            with st.spinner("Configurando SESCA..."):
                ok = setup_sesca(log=lambda msg: st.text(msg))
                st.session_state["sesca_setup_done"] = ok
            if ok:
                st.rerun()
            else:
                st.error("Falha na configuracao. Verifique os logs.")

    st.markdown("---")

    # Basis set
    basis = st.selectbox(
        "Conjunto de base espectral",
        options=BASIS_OPTIONS,
        index=BASIS_OPTIONS.index(DEFAULT_BASIS),
        help="DS-dT e recomendado para proteinas globulares. "
             "DS5-4 usa 5 componentes. DSSP-1 e HBSS-3 usam atribuicoes DSSP.",
    )

    # Limpar PDB
    clean = st.toggle(
        "Limpar PDB antes de processar",
        value=True,
        help="Remove HETATM (agua, ligantes), conformacoes alternativas, "
             "e mantem apenas o primeiro modelo NMR.",
    )

    st.markdown("---")

    # Referencia
    st.markdown(
        "<small style='color: #9ca3af;'>"
        "<b>Referencia:</b><br>"
        "Nagy et al., J. Chem. Theory Comput. 15, 5087-5102 (2019)<br>"
        "<a href='https://doi.org/10.1021/acs.jctc.9b00203' style='color: #6b7280;'>"
        "doi: 10.1021/acs.jctc.9b00203</a>"
        "</small>",
        unsafe_allow_html=True,
    )


# =============================================================================
#  HEADER
# =============================================================================

st.markdown("""
<div class="main-header">
    <h1>SESCA &mdash; CD Spectrum Predictor</h1>
    <p>Predicao de espectros de Dicroismo Circular a partir de estruturas proteicas (PDB)</p>
</div>
""", unsafe_allow_html=True)


# =============================================================================
#  INPUT DE ESTRUTURAS
# =============================================================================

tab_upload, tab_rcsb = st.tabs(["Upload de PDB", "Buscar no RCSB PDB"])

uploaded_files = []
pdb_ids = []

with tab_upload:
    files = st.file_uploader(
        "Arraste seus arquivos PDB aqui",
        type=["pdb"],
        accept_multiple_files=True,
        help="Aceita um ou mais arquivos .pdb",
    )
    if files:
        uploaded_files = files

with tab_rcsb:
    col1, col2 = st.columns([3, 1])
    with col1:
        pdb_input = st.text_input(
            "Codigos PDB (separados por espaco ou virgula)",
            placeholder="ex: 1UBQ 2GB1 1L2Y",
            help="Digite os codigos de acesso do RCSB PDB",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        fetch_btn = st.button("Buscar", use_container_width=True, type="secondary")

    if pdb_input:
        pdb_ids = [x.strip().upper() for x in pdb_input.replace(",", " ").split() if x.strip()]
        if pdb_ids:
            st.info(f"Estruturas selecionadas: **{', '.join(pdb_ids)}**")


# =============================================================================
#  EXECUCAO
# =============================================================================

has_input = bool(uploaded_files) or bool(pdb_ids)

run_col1, run_col2, run_col3 = st.columns([1, 2, 1])
with run_col2:
    run_btn = st.button(
        "Executar Predicao",
        use_container_width=True,
        type="primary",
        disabled=not (has_input and sesca_ready),
    )

if not sesca_ready and has_input:
    st.warning("Configure o SESCA primeiro usando o botao na barra lateral.")

if run_btn and has_input and sesca_ready:
    st.markdown("---")

    results = {}
    logs = []

    def log_msg(msg):
        logs.append(msg)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        pdb_dir = tmpdir / "pdb_inputs"
        pdb_dir.mkdir()

        progress = st.progress(0, text="Preparando...")

        # Reune todos os PDBs
        all_pdbs = []

        # Arquivos enviados por upload
        for uf in uploaded_files:
            dest = pdb_dir / uf.name
            dest.write_bytes(uf.read())
            all_pdbs.append(dest)

        # PDBs baixados do RCSB
        for i, pid in enumerate(pdb_ids):
            progress.progress(
                int(10 + 20 * i / max(len(pdb_ids), 1)),
                text=f"Baixando {pid}...",
            )
            path = fetch_pdb(pid, pdb_dir, log=log_msg)
            if path:
                all_pdbs.append(path)

        if not all_pdbs:
            st.error("Nenhum arquivo PDB valido para processar.")
            st.stop()

        # Processa cada PDB
        total = len(all_pdbs)
        for i, pdb_path in enumerate(all_pdbs):
            pct = int(30 + 60 * i / total)
            progress.progress(pct, text=f"Processando {pdb_path.stem}...")

            pdb_to_use = clean_pdb(pdb_path, tmpdir) if clean else pdb_path
            result = run_sesca(pdb_to_use, tmpdir, basis=basis, log=log_msg)

            if result:
                results[pdb_path.stem] = result

        progress.progress(95, text="Finalizando...")

        # Salva CSV combinado
        if results:
            csv_path = save_combined_csv(results, tmpdir)
            csv_data = csv_path.read_text()

        progress.progress(100, text="Concluido!")

    # -- Exibe logs
    with st.expander("Log de execucao", expanded=False):
        for line in logs:
            st.text(line)

    if not results:
        st.error("Nenhum espectro foi gerado. Verifique os logs acima.")
        st.stop()

    # -- Armazena em session_state para persistir
    st.session_state["results"] = results
    st.session_state["csv_data"] = csv_data


# =============================================================================
#  RESULTADOS
# =============================================================================

if "results" in st.session_state and st.session_state["results"]:
    results = st.session_state["results"]
    csv_data = st.session_state.get("csv_data", "")

    st.markdown("---")
    st.markdown("## Resultados")

    # -- Metricas resumo
    cols = st.columns(len(results))
    palette = ["#4b5563", "#1f2937", "#374151", "#6b7280", "#111827", "#525252", "#3f3f46"]

    for i, (name, r) in enumerate(results.items()):
        cd = r["cd_values"]
        wl = r["wavelengths"]
        i_min = cd.index(min(cd))
        i_max = cd.index(max(cd))

        with cols[i]:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="label">{name}</div>'
                f'<div class="value">{min(cd):.2f}</div>'
                f'<div class="unit">Min. CD @ {wl[i_min]:.0f} nm</div>'
                f'<br>'
                f'<div class="value">{max(cd):.2f}</div>'
                f'<div class="unit">Max. CD @ {wl[i_max]:.0f} nm</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # -- Grafico interativo (Plotly)
    fig = go.Figure()

    # Paleta suave e profissional
    line_colors = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2", "#be185d"]

    for i, (name, r) in enumerate(results.items()):
        color = line_colors[i % len(line_colors)]
        fig.add_trace(go.Scatter(
            x=r["wavelengths"],
            y=r["cd_values"],
            name=name,
            mode="lines",
            line=dict(color=color, width=2),
            hovertemplate="<b>%{fullData.name}</b><br>"
                          "lambda = %{x:.1f} nm<br>"
                          "CD = %{y:.4f}<extra></extra>",
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="rgba(0, 0, 0, 0.15)")

    fig.update_layout(
        template="simple_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        title=dict(
            text="Espectros de Dicroismo Circular Preditos",
            font=dict(size=16, color="#111827", family="sans-serif"),
        ),
        xaxis=dict(
            title="Comprimento de onda (nm)",
            gridcolor="#f3f4f6",
            linecolor="#d1d5db",
            dtick=10,
            title_font=dict(size=13, color="#374151"),
            tickfont=dict(size=11, color="#6b7280"),
        ),
        yaxis=dict(
            title="CD (delta-epsilon)",
            gridcolor="#f3f4f6",
            linecolor="#d1d5db",
            title_font=dict(size=13, color="#374151"),
            tickfont=dict(size=11, color="#6b7280"),
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#e5e7eb",
            borderwidth=1,
            font=dict(color="#374151", size=12),
        ),
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor="#d1d5db",
            font=dict(color="#111827", size=12),
        ),
        height=480,
        margin=dict(l=60, r=30, t=50, b=60),
    )

    st.plotly_chart(fig, use_container_width=True)

    # -- Tabela de dados
    with st.expander("Tabela de dados", expanded=False):
        all_wl = sorted({wl for r in results.values() for wl in r["wavelengths"]})
        df_data = {"Wavelength (nm)": all_wl}
        for name, r in results.items():
            wl_map = dict(zip(r["wavelengths"], r["cd_values"]))
            df_data[name] = [wl_map.get(wl, None) for wl in all_wl]
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # -- Downloads
    st.markdown("### Downloads")
    dl_col1, dl_col2, dl_col3 = st.columns(3)

    with dl_col1:
        st.download_button(
            "CSV Combinado",
            data=csv_data,
            file_name="espectros_CD_combinados.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with dl_col2:
        html_buf = io.StringIO()
        fig.write_html(html_buf, include_plotlyjs="cdn")
        st.download_button(
            "Grafico Interativo (HTML)",
            data=html_buf.getvalue(),
            file_name="espectro_CD_interativo.html",
            mime="text/html",
            use_container_width=True,
        )

    with dl_col3:
        try:
            png_bytes = fig.to_image(format="png", width=1200, height=600, scale=2)
            st.download_button(
                "Grafico (PNG)",
                data=png_bytes,
                file_name="espectro_CD.png",
                mime="image/png",
                use_container_width=True,
            )
        except Exception:
            st.caption("PNG requer kaleido: pip install kaleido")

else:
    # Estado vazio
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; padding: 3rem; color: #9ca3af;'>"
        "<p style='font-size: 1rem;'>Envie um arquivo PDB ou busque pelo codigo RCSB para comecar</p>"
        "</div>",
        unsafe_allow_html=True,
    )
