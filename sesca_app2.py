#!/usr/bin/env python3
"""
SESCA Web App - Predicao de Espectros de Dicroismo Circular (CD)
Executar com: streamlit run sesca_app.py
"""

import tempfile
import io
import os
import sys
import subprocess
import zipfile
import urllib.request
import csv
from pathlib import Path

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd


# =============================================================================
#  CONFIGURACOES GLOBAIS
# =============================================================================

APP_DIR = Path(__file__).resolve().parent
SESCA_URL = "https://www.mpinat.mpg.de/4787098/SESCA_v097.zip"
SESCA_DIR = APP_DIR / "SESCA_v097"
SESCA_MAIN = SESCA_DIR / "scripts" / "SESCA_main.py"
SESCA_SETUP = SESCA_DIR / "setup.py"

DEFAULT_BASIS = "DS-dT"
BASIS_OPTIONS = ["DS-dT", "DS5-4", "DSSP-1", "HBSS-3", "BestSel_der"]


# =============================================================================
#  FUNCOES CORE
# =============================================================================

def install_sesca(log=print):
    """Baixa, descompacta e configura o SESCA."""
    zip_path = APP_DIR / "SESCA_v097.zip"

    if SESCA_MAIN.exists():
        log("SESCA ja esta instalado.")
        return True

    # Baixa o zip
    log("Baixando SESCA (~5 MB)...")
    try:
        urllib.request.urlretrieve(SESCA_URL, zip_path)
    except Exception as e:
        log(f"Falha no download: {e}")
        return False

    # Descompacta
    log("Descompactando...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(str(APP_DIR))
        zip_path.unlink(missing_ok=True)
    except Exception as e:
        log(f"Falha ao descompactar: {e}")
        return False

    if not SESCA_SETUP.exists():
        log("Erro: setup.py nao encontrado apos descompactar.")
        return False

    # Roda o setup.py para configurar caminhos internos
    log("Configurando caminhos internos...")
    try:
        result = subprocess.run(
            [sys.executable, str(SESCA_SETUP)],
            capture_output=True, text=True,
            cwd=str(SESCA_DIR),
        )
        if result.returncode != 0:
            log(f"Erro no setup: {result.stderr[-500:]}")
            return False
    except Exception as e:
        log(f"Falha ao rodar setup: {e}")
        return False

    # Remove exemplos para economizar espaco
    examples_dir = SESCA_DIR / "examples"
    if examples_dir.exists():
        import shutil
        shutil.rmtree(examples_dir, ignore_errors=True)

    log("SESCA instalado e configurado com sucesso.")
    return True


def check_sesca():
    """Verifica se o SESCA esta disponivel."""
    return SESCA_MAIN.exists()


def fetch_pdb(pdb_id, output_dir, log=print):
    """Baixa um arquivo PDB do RCSB."""
    pdb_id = pdb_id.upper()
    dest = output_dir / f"{pdb_id}.pdb"

    if dest.exists():
        log(f"{pdb_id}.pdb ja existe.")
        return dest

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    log(f"Baixando {pdb_id} do RCSB PDB...")

    try:
        urllib.request.urlretrieve(url, dest)
        log(f"Salvo: {dest.name}")
        return dest
    except Exception as e:
        log(f"Nao foi possivel baixar {pdb_id}: {e}")
        return None


def clean_pdb(input_pdb, output_dir):
    """Remove HETATM, conformacoes alternativas, mantem 1o modelo."""
    clean_path = output_dir / f"{input_pdb.stem}_clean.pdb"

    with open(input_pdb) as fh, open(clean_path, "w") as out:
        model_done = False
        for line in fh:
            record = line[:6].strip()
            if record == "MODEL":
                if model_done:
                    break
                continue
            if record == "ENDMDL":
                model_done = True
                continue
            if record == "HETATM":
                continue
            if record == "ATOM":
                if line[16] not in (" ", "A"):
                    continue
            out.write(line)

    return clean_path


def run_sesca(pdb_path, output_dir, basis=DEFAULT_BASIS, log=print):
    """Executa SESCA_main.py para um PDB."""
    out_file = output_dir / f"{pdb_path.stem}_CDspectrum.dat"

    cmd = [
        sys.executable, str(SESCA_MAIN),
        "@pdb",   str(pdb_path),
        "@lib",   basis,
        "@write", str(out_file),
    ]

    log(f"Processando: {pdb_path.name} (base: {basis})")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=str(SESCA_DIR),
        )
    except FileNotFoundError:
        log(f"Erro: nao foi possivel executar Python.")
        return None

    if result.returncode != 0:
        err = result.stderr or result.stdout or "Erro desconhecido"
        log(f"SESCA erro (codigo {result.returncode}): {err[-1000:]}")
        return None

    if result.stdout:
        for line in result.stdout.splitlines():
            if any(kw in line.lower() for kw in ["helix", "sheet", "rmsd", "error", "warning", "spectrum"]):
                log(f"  {line.strip()}")

    return parse_sesca_output(out_file, log=log)


def parse_sesca_output(dat_file, log=print):
    """Le o arquivo .dat gerado pelo SESCA."""
    if not dat_file.exists():
        log(f"Arquivo de saida nao encontrado: {dat_file.name}")
        return None

    wavelengths, cd_values = [], []
    with open(dat_file) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    wavelengths.append(float(parts[0]))
                    cd_values.append(float(parts[1]))
                except ValueError:
                    continue

    if not wavelengths:
        log(f"Nenhum dado lido de {dat_file.name}")
        return None

    log(f"Lidos {len(wavelengths)} pontos ({wavelengths[0]:.0f}-{wavelengths[-1]:.0f} nm)")

    return {
        "file": str(dat_file),
        "wavelengths": wavelengths,
        "cd_values": cd_values,
    }


def save_combined_csv(results, output_dir):
    """Salva todos os espectros em um CSV."""
    csv_path = output_dir / "espectros_CD_combinados.csv"
    all_wl = sorted({wl for r in results.values() for wl in r["wavelengths"]})
    names = list(results.keys())

    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Wavelength_nm"] + names)
        for wl in all_wl:
            row = [wl]
            for name in names:
                r = results[name]
                try:
                    idx = r["wavelengths"].index(wl)
                    row.append(r["cd_values"][idx])
                except ValueError:
                    row.append("")
            writer.writerow(row)

    return csv_path


# =============================================================================
#  CONFIGURACAO DA PAGINA
# =============================================================================

st.set_page_config(
    page_title="SESCA | CD Spectrum Predictor",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
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
    .status-ok { color: #059669; font-size: 0.85rem; font-weight: 500; }
    .status-err { color: #dc2626; font-size: 0.85rem; font-weight: 500; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] { border-radius: 6px 6px 0 0; padding: 8px 20px; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
#  SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("## Configuracoes")
    st.markdown("---")

    # Instalacao automatica do SESCA
    if "sesca_ready" not in st.session_state:
        if check_sesca():
            st.session_state["sesca_ready"] = True
        else:
            with st.spinner("Instalando SESCA (primeira execucao)..."):
                install_logs = []
                ok = install_sesca(log=lambda msg: install_logs.append(msg))
                st.session_state["sesca_ready"] = ok
                if not ok:
                    for line in install_logs:
                        st.text(line)

    sesca_ready = st.session_state.get("sesca_ready", False)
    if sesca_ready:
        st.markdown('<span class="status-ok">SESCA pronto</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-err">SESCA nao disponivel</span>', unsafe_allow_html=True)
        if st.button("Tentar novamente", use_container_width=True):
            del st.session_state["sesca_ready"]
            st.rerun()

    st.markdown("---")

    basis = st.selectbox(
        "Conjunto de base espectral",
        options=BASIS_OPTIONS,
        index=BASIS_OPTIONS.index(DEFAULT_BASIS),
        help="DS-dT e recomendado para proteinas globulares.",
    )

    clean = st.toggle(
        "Limpar PDB antes de processar",
        value=True,
        help="Remove HETATM (agua, ligantes), conformacoes alternativas, "
             "e mantem apenas o primeiro modelo NMR.",
    )

    st.markdown("---")
    st.markdown(
        "<small style='color: #9ca3af;'>"
        "<b>Referencia:</b><br>"
        "Nagy et al., JCTC 15, 5087-5102 (2019)<br>"
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
#  INPUT
# =============================================================================

tab_upload, tab_rcsb = st.tabs(["Upload de PDB", "Buscar no RCSB PDB"])

uploaded_files = []
pdb_ids = []

with tab_upload:
    files = st.file_uploader(
        "Arraste seus arquivos PDB aqui",
        type=["pdb"],
        accept_multiple_files=True,
    )
    if files:
        uploaded_files = files

with tab_rcsb:
    col1, col2 = st.columns([3, 1])
    with col1:
        pdb_input = st.text_input(
            "Codigos PDB (separados por espaco ou virgula)",
            placeholder="ex: 1UBQ 2GB1 1L2Y",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("Buscar", use_container_width=True, type="secondary")

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
    st.warning("Aguarde a instalacao do SESCA ou clique em 'Tentar novamente'.")

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

        all_pdbs = []

        for uf in uploaded_files:
            dest = pdb_dir / uf.name
            dest.write_bytes(uf.read())
            all_pdbs.append(dest)

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

        total = len(all_pdbs)
        for i, pdb_path in enumerate(all_pdbs):
            pct = int(30 + 60 * i / total)
            progress.progress(pct, text=f"Processando {pdb_path.stem}...")

            pdb_to_use = clean_pdb(pdb_path, tmpdir) if clean else pdb_path
            result = run_sesca(pdb_to_use, tmpdir, basis=basis, log=log_msg)

            if result:
                results[pdb_path.stem] = result

        progress.progress(95, text="Finalizando...")

        if results:
            csv_path = save_combined_csv(results, tmpdir)
            csv_data = csv_path.read_text()

        progress.progress(100, text="Concluido!")

    with st.expander("Log de execucao", expanded=False):
        for line in logs:
            st.text(line)

    if not results:
        st.error("Nenhum espectro foi gerado. Verifique os logs acima.")
        st.stop()

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

    cols = st.columns(len(results))
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

    fig = go.Figure()
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
            font=dict(size=16, color="#111827"),
        ),
        xaxis=dict(
            title="Comprimento de onda (nm)",
            gridcolor="#f3f4f6", linecolor="#d1d5db", dtick=10,
            title_font=dict(size=13, color="#374151"),
            tickfont=dict(size=11, color="#6b7280"),
        ),
        yaxis=dict(
            title="CD (delta-epsilon)",
            gridcolor="#f3f4f6", linecolor="#d1d5db",
            title_font=dict(size=13, color="#374151"),
            tickfont=dict(size=11, color="#6b7280"),
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#e5e7eb", borderwidth=1,
            font=dict(color="#374151", size=12),
        ),
        hoverlabel=dict(
            bgcolor="#ffffff", bordercolor="#d1d5db",
            font=dict(color="#111827", size=12),
        ),
        height=480,
        margin=dict(l=60, r=30, t=50, b=60),
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Tabela de dados", expanded=False):
        all_wl = sorted({wl for r in results.values() for wl in r["wavelengths"]})
        df_data = {"Wavelength (nm)": all_wl}
        for name, r in results.items():
            wl_map = dict(zip(r["wavelengths"], r["cd_values"]))
            df_data[name] = [wl_map.get(wl, None) for wl in all_wl]
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

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
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; padding: 3rem; color: #9ca3af;'>"
        "<p style='font-size: 1rem;'>Envie um arquivo PDB ou busque pelo codigo RCSB para comecar</p>"
        "</div>",
        unsafe_allow_html=True,
    )
