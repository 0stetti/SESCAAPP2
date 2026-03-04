#!/usr/bin/env python3
"""
=============================================================================
  SESCA Pipeline - Predição de Espectros de Dicroísmo Circular (CD)
=============================================================================

O que este script faz:
  1. Baixa e descompacta o pacote SESCA (Max Planck Institute)
  2. Aceita um ou mais arquivos PDB como entrada
  3. Roda a predição de espectro de CD para cada proteína
  4. Salva os resultados em CSV e gera um gráfico comparativo

Como usar:
  python sesca_pipeline.py --pdb minha_proteina.pdb
  python sesca_pipeline.py --pdb prot1.pdb prot2.pdb --output resultados/
  python sesca_pipeline.py --pdb_id 1UBQ           # baixa do RCSB PDB
  python sesca_pipeline.py --pdb_id 1UBQ 2GB1 --plot

Dependências (instalar com pip):
  pip install numpy matplotlib requests biopython

Referência:
  Nagy et al., J. Chem. Theory Comput. 15, 5087-5102 (2019)
  https://doi.org/10.1021/acs.jctc.9b00203
=============================================================================
"""

import os
import sys
import argparse
import zipfile
import subprocess
import urllib.request
import shutil
import csv
from pathlib import Path

# ─── Dependências opcionais ───────────────────────────────────────────────────
try:
    import numpy as np
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("[AVISO] matplotlib/numpy não encontrado. Gráficos desativados.")
    print("        Instale com: pip install numpy matplotlib")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# =============================================================================
#  CONFIGURAÇÕES GLOBAIS
# =============================================================================

SESCA_URL   = "https://www.mpinat.mpg.de/4787098/SESCA_v097.zip"
SESCA_DIR   = Path("SESCA_v097")          # pasta após descompactar
SESCA_MAIN  = SESCA_DIR / "SESCA_main.py" # script principal do SESCA

# Conjunto de base padrão (melhor custo-benefício para proteínas globulares)
# Outros disponíveis no pacote: DS5-4, DSSP-1, HBSS-3, BestSel_der
DEFAULT_BASIS = "DS-dT"

# Intervalo de comprimento de onda padrão do SESCA (nm)
WL_MIN, WL_MAX = 175, 250


# =============================================================================
#  1. INSTALAÇÃO DO SESCA
# =============================================================================

def download_sesca(force=False):
    """Baixa e descompacta o SESCA se ainda não estiver presente."""
    zip_path = Path("SESCA_v097.zip")

    if SESCA_DIR.exists() and not force:
        print(f"[OK] SESCA já está em '{SESCA_DIR}'. Pulando download.")
        return True

    print(f"[INFO] Baixando SESCA de:\n       {SESCA_URL}")
    print("       (arquivo ~5 MB, pode demorar alguns segundos)\n")

    try:
        urllib.request.urlretrieve(SESCA_URL, zip_path)
    except Exception as e:
        print(f"[ERRO] Falha no download: {e}")
        print("\n  → Baixe manualmente em: https://www.mpinat.mpg.de/sesca")
        print(  "    e descompacte na mesma pasta deste script.")
        return False

    print("[INFO] Descompactando...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(".")
    zip_path.unlink()  # remove o zip após extrair

    if not SESCA_MAIN.exists():
        print(f"[ERRO] Não encontrei '{SESCA_MAIN}' após descompactar.")
        print("       Verifique o conteúdo da pasta SESCA_v097/")
        return False

    print(f"[OK] SESCA instalado em '{SESCA_DIR}'\n")
    return True


def check_sesca():
    """Verifica se o SESCA está disponível e funcional."""
    if not SESCA_MAIN.exists():
        print("[ERRO] SESCA não encontrado. Execute com --setup primeiro.")
        return False

    # Verifica Python e dependências do SESCA (numpy obrigatório)
    try:
        import numpy  # noqa: F401
    except ImportError:
        print("[ERRO] numpy é obrigatório para o SESCA.")
        print("       Instale com: pip install numpy")
        return False

    return True


# =============================================================================
#  2. DOWNLOAD DE ESTRUTURAS DO RCSB PDB
# =============================================================================

def fetch_pdb(pdb_id: str, output_dir: Path) -> Path | None:
    """Baixa um arquivo PDB do RCSB pelo código de acesso (ex: 1UBQ)."""
    pdb_id = pdb_id.upper()
    dest   = output_dir / f"{pdb_id}.pdb"

    if dest.exists():
        print(f"[OK] {pdb_id}.pdb já existe localmente.")
        return dest

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"[INFO] Baixando estrutura {pdb_id} do RCSB PDB...")

    try:
        urllib.request.urlretrieve(url, dest)
        print(f"[OK] Salvo em {dest}")
        return dest
    except Exception as e:
        print(f"[ERRO] Não foi possível baixar {pdb_id}: {e}")
        return None


# =============================================================================
#  3. PRÉ-PROCESSAMENTO DO PDB
# =============================================================================

def clean_pdb(input_pdb: Path, output_dir: Path) -> Path:
    """
    Limpeza básica do PDB:
    - Remove linhas HETATM (ligantes, água)
    - Mantém apenas o primeiro modelo (para NMR multi-modelo)
    - Remove cadeias alternativas (mantém conf. A)
    Retorna o caminho do PDB limpo.
    """
    clean_path = output_dir / f"{input_pdb.stem}_clean.pdb"

    with open(input_pdb) as fh, open(clean_path, "w") as out:
        in_model = False
        model_done = False

        for line in fh:
            record = line[:6].strip()

            # Para NMR: processa só o primeiro MODEL
            if record == "MODEL":
                if model_done:
                    break
                in_model = True
                continue
            if record == "ENDMDL":
                model_done = True
                in_model = False
                continue

            # Pula HETATM (água, ligantes, cofatores)
            if record == "HETATM":
                continue

            # Mantém apenas conformação A (ou sem indicação)
            if record == "ATOM":
                alt_loc = line[16]
                if alt_loc not in (" ", "A"):
                    continue

            out.write(line)

    return clean_path


# =============================================================================
#  4. EXECUÇÃO DO SESCA
# =============================================================================

def run_sesca(pdb_path: Path, output_dir: Path, basis: str = DEFAULT_BASIS) -> dict | None:
    """
    Executa o SESCA_main.py para um arquivo PDB.

    O SESCA recebe:
      -pdb   : arquivo de estrutura (.pdb)
      -basis : conjunto de espectros base (padrão: DS-dT)
      -ofile : arquivo de saída com o espectro predito

    Retorna dicionário com arrays de comprimento de onda e intensidade CD.
    """
    out_file = output_dir / f"{pdb_path.stem}_CDspectrum.dat"

    cmd = [
        sys.executable, str(SESCA_MAIN),
        "-pdb",   str(pdb_path),
        "-basis", basis,
        "-ofile", str(out_file),
    ]

    print(f"\n[SESCA] Processando: {pdb_path.name}")
    print(f"        Conjunto de base: {basis}")
    print(f"        Saída: {out_file}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(SESCA_DIR.parent),
        )
    except FileNotFoundError:
        print(f"[ERRO] Não foi possível executar: {cmd[0]}")
        return None

    if result.returncode != 0:
        print(f"[ERRO] SESCA retornou erro (código {result.returncode}):")
        print(result.stderr[-2000:])  # últimas 2000 chars do erro
        return None

    if result.stdout:
        # Exibe as linhas mais relevantes da saída
        for line in result.stdout.splitlines():
            if any(kw in line.lower() for kw in ["helix", "sheet", "rmsd", "error", "warning", "%"]):
                print(f"        {line}")

    # Lê o espectro predito
    return parse_sesca_output(out_file)


def parse_sesca_output(dat_file: Path) -> dict | None:
    """
    Lê o arquivo .dat gerado pelo SESCA.

    Formato típico (2 colunas):
      # Wavelength(nm)  CD(Δε ou MRE)
      175.0   -4.231
      176.0   -3.890
      ...
    """
    if not dat_file.exists():
        print(f"[ERRO] Arquivo de saída não encontrado: {dat_file}")
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
        print(f"[ERRO] Nenhum dado lido de {dat_file}")
        return None

    print(f"[OK]   Lidos {len(wavelengths)} pontos espectrais "
          f"({wavelengths[0]:.0f}–{wavelengths[-1]:.0f} nm)")

    return {
        "file":        str(dat_file),
        "wavelengths": wavelengths,
        "cd_values":   cd_values,
    }


# =============================================================================
#  5. SALVAMENTO DOS RESULTADOS
# =============================================================================

def save_combined_csv(results: dict, output_dir: Path):
    """Salva todos os espectros em um único CSV para análise posterior."""
    csv_path = output_dir / "espectros_CD_combinados.csv"

    # Coleta todos os comprimentos de onda únicos
    all_wl = sorted({wl for r in results.values() for wl in r["wavelengths"]})
    names  = list(results.keys())

    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Comprimento_de_onda_nm"] + names)

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

    print(f"\n[OK] CSV combinado salvo: {csv_path}")
    return csv_path


def save_summary_txt(results: dict, output_dir: Path):
    """Salva resumo numérico dos espectros (máximo, mínimo, cruzamento zero)."""
    summary_path = output_dir / "resumo_espectros.txt"

    with open(summary_path, "w") as fh:
        fh.write("=" * 60 + "\n")
        fh.write("  Resumo dos Espectros de CD Preditos (SESCA)\n")
        fh.write("=" * 60 + "\n\n")

        for name, r in results.items():
            wl  = r["wavelengths"]
            cd  = r["cd_values"]
            i_min = cd.index(min(cd))
            i_max = cd.index(max(cd))

            fh.write(f"Proteína: {name}\n")
            fh.write(f"  Pontos espectrais : {len(wl)}\n")
            fh.write(f"  Intervalo (nm)    : {wl[0]:.0f} – {wl[-1]:.0f}\n")
            fh.write(f"  Mínimo CD         : {min(cd):.4f}  @ {wl[i_min]:.0f} nm\n")
            fh.write(f"  Máximo CD         : {max(cd):.4f}  @ {wl[i_max]:.0f} nm\n")
            fh.write("\n")

    print(f"[OK] Resumo salvo: {summary_path}")
    return summary_path


# =============================================================================
#  6. VISUALIZAÇÃO
# =============================================================================

def plot_spectra(results: dict, output_dir: Path):
    """Gera gráfico de todos os espectros CD sobrepostos."""
    if not HAS_PLOT:
        print("[AVISO] matplotlib não disponível — gráfico não gerado.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    colors = plt.cm.tab10.colors
    for i, (name, r) in enumerate(results.items()):
        ax.plot(
            r["wavelengths"],
            r["cd_values"],
            label=name,
            color=colors[i % len(colors)],
            linewidth=2,
        )

    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
    ax.set_xlabel("Comprimento de onda (nm)", fontsize=12)
    ax.set_ylabel("CD (Δε ou MRE)", fontsize=12)
    ax.set_title("Espectros de Dicroísmo Circular Preditos (SESCA)", fontsize=13)
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = output_dir / "espectros_CD.png"
    fig.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[OK] Gráfico salvo: {plot_path}")


# =============================================================================
#  7. PIPELINE PRINCIPAL
# =============================================================================

def run_pipeline(pdb_files: list[Path],
                 pdb_ids: list[str],
                 output_dir: Path,
                 basis: str,
                 plot: bool,
                 clean: bool):
    """Orquestra todo o pipeline para uma lista de PDBs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir = output_dir / "pdb_inputs"
    pdb_dir.mkdir(exist_ok=True)

    # ── Baixa estruturas pelo ID se solicitado ─────────────────────────────
    for pdb_id in pdb_ids:
        path = fetch_pdb(pdb_id, pdb_dir)
        if path:
            pdb_files.append(path)

    if not pdb_files:
        print("[ERRO] Nenhum arquivo PDB fornecido ou baixado com sucesso.")
        sys.exit(1)

    # ── Processa cada PDB ──────────────────────────────────────────────────
    results = {}

    for pdb_path in pdb_files:
        pdb_path = Path(pdb_path)

        if not pdb_path.exists():
            print(f"[AVISO] Arquivo não encontrado: {pdb_path}. Pulando.")
            continue

        # Limpeza opcional
        if clean:
            pdb_to_use = clean_pdb(pdb_path, output_dir)
            print(f"[INFO] PDB limpo: {pdb_to_use}")
        else:
            pdb_to_use = pdb_path

        # Predição SESCA
        result = run_sesca(pdb_to_use, output_dir, basis=basis)

        if result:
            results[pdb_path.stem] = result

    if not results:
        print("\n[ERRO] Nenhum espectro foi gerado.")
        sys.exit(1)

    # ── Salva resultados ───────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  Salvando resultados...")
    print("=" * 50)

    save_combined_csv(results, output_dir)
    save_summary_txt(results, output_dir)

    if plot:
        plot_spectra(results, output_dir)

    print(f"\n✅ Pipeline concluído! Resultados em: {output_dir.resolve()}\n")
    print("  Arquivos gerados:")
    for f in sorted(output_dir.glob("*")):
        if f.is_file():
            print(f"    {f.name}")


# =============================================================================
#  8. INTERFACE DE LINHA DE COMANDO
# =============================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="Pipeline SESCA: predição de espectros de CD a partir de PDBs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Prediz espectro de um PDB local:
  python sesca_pipeline.py --pdb minha_proteina.pdb

  # Baixa do RCSB e prediz, com gráfico:
  python sesca_pipeline.py --pdb_id 1UBQ --plot

  # Múltiplas proteínas comparadas:
  python sesca_pipeline.py --pdb_id 1UBQ 2GB1 1L2Y --plot --output resultados/

  # Só instalar o SESCA (sem processar nada):
  python sesca_pipeline.py --setup
        """
    )

    parser.add_argument(
        "--pdb", nargs="+", type=Path, default=[],
        help="Um ou mais arquivos PDB locais",
    )
    parser.add_argument(
        "--pdb_id", nargs="+", default=[],
        metavar="ID",
        help="Código(s) PDB do RCSB para baixar automaticamente (ex: 1UBQ)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("sesca_resultados"),
        help="Pasta de saída (padrão: sesca_resultados/)",
    )
    parser.add_argument(
        "--basis", default=DEFAULT_BASIS,
        choices=["DS-dT", "DS5-4", "DSSP-1", "HBSS-3", "BestSel_der"],
        help=f"Conjunto de base espectral (padrão: {DEFAULT_BASIS})",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Gera gráfico PNG com os espectros",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Remove HETATM e cadeias alternativas antes de rodar o SESCA",
    )
    parser.add_argument(
        "--setup", action="store_true",
        help="Apenas baixa e instala o SESCA, sem processar PDBs",
    )
    parser.add_argument(
        "--force_download", action="store_true",
        help="Força re-download do SESCA mesmo se já instalado",
    )

    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()

    print("\n" + "=" * 60)
    print("  SESCA Pipeline — Predição de Espectros de CD")
    print("=" * 60 + "\n")

    # ── Instalação ────────────────────────────────────────────────────────
    ok = download_sesca(force=args.force_download)
    if not ok:
        sys.exit(1)

    if args.setup:
        print("[OK] Setup concluído. SESCA pronto para uso.")
        sys.exit(0)

    if not check_sesca():
        sys.exit(1)

    # ── Pipeline ──────────────────────────────────────────────────────────
    if not args.pdb and not args.pdb_id:
        print("[ERRO] Forneça pelo menos um PDB com --pdb ou --pdb_id.")
        print("       Use --help para ver exemplos.")
        parser.print_help()
        sys.exit(1)

    run_pipeline(
        pdb_files  = list(args.pdb),
        pdb_ids    = args.pdb_id,
        output_dir = args.output,
        basis      = args.basis,
        plot       = args.plot,
        clean      = args.clean,
    )


if __name__ == "__main__":
    main()