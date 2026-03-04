#!/usr/bin/env python3
"""
=============================================================================
  SESCA Pipeline CLI - Predição de Espectros de Dicroísmo Circular (CD)
=============================================================================

Como usar:
  python sesca_pipeline.py --pdb minha_proteina.pdb
  python sesca_pipeline.py --pdb prot1.pdb prot2.pdb --output resultados/
  python sesca_pipeline.py --pdb_id 1UBQ           # baixa do RCSB PDB
  python sesca_pipeline.py --pdb_id 1UBQ 2GB1 --plot
"""

import sys
import argparse
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

from sesca_core import (
    download_sesca, check_sesca, fetch_pdb, clean_pdb,
    run_sesca, save_combined_csv, save_summary_txt,
    DEFAULT_BASIS, BASIS_OPTIONS,
)


def plot_spectra(results: dict, output_dir: Path):
    """Gera gráfico de todos os espectros CD sobrepostos."""
    if not HAS_PLOT:
        print("[AVISO] matplotlib não disponível — gráfico não gerado.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10.colors

    for i, (name, r) in enumerate(results.items()):
        ax.plot(
            r["wavelengths"], r["cd_values"],
            label=name, color=colors[i % len(colors)], linewidth=2,
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


def run_pipeline(pdb_files, pdb_ids, output_dir, basis, plot, clean):
    """Orquestra todo o pipeline para uma lista de PDBs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir = output_dir / "pdb_inputs"
    pdb_dir.mkdir(exist_ok=True)

    for pdb_id in pdb_ids:
        path = fetch_pdb(pdb_id, pdb_dir)
        if path:
            pdb_files.append(path)

    if not pdb_files:
        print("[ERRO] Nenhum arquivo PDB fornecido ou baixado com sucesso.")
        sys.exit(1)

    results = {}
    for pdb_path in pdb_files:
        pdb_path = Path(pdb_path)
        if not pdb_path.exists():
            print(f"[AVISO] Arquivo não encontrado: {pdb_path}. Pulando.")
            continue

        pdb_to_use = clean_pdb(pdb_path, output_dir) if clean else pdb_path
        result = run_sesca(pdb_to_use, output_dir, basis=basis)
        if result:
            results[pdb_path.stem] = result

    if not results:
        print("\n[ERRO] Nenhum espectro foi gerado.")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("  Salvando resultados...")
    print("=" * 50)

    save_combined_csv(results, output_dir)
    save_summary_txt(results, output_dir)

    if plot:
        plot_spectra(results, output_dir)

    print(f"\nPipeline concluído! Resultados em: {output_dir.resolve()}\n")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Pipeline SESCA: predição de espectros de CD a partir de PDBs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python sesca_pipeline.py --pdb minha_proteina.pdb
  python sesca_pipeline.py --pdb_id 1UBQ --plot
  python sesca_pipeline.py --pdb_id 1UBQ 2GB1 1L2Y --plot --output resultados/
  python sesca_pipeline.py --setup
        """
    )
    parser.add_argument("--pdb", nargs="+", type=Path, default=[])
    parser.add_argument("--pdb_id", nargs="+", default=[], metavar="ID")
    parser.add_argument("--output", type=Path, default=Path("sesca_resultados"))
    parser.add_argument("--basis", default=DEFAULT_BASIS, choices=BASIS_OPTIONS)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--force_download", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  SESCA Pipeline — Predição de Espectros de CD")
    print("=" * 60 + "\n")

    ok = download_sesca(force=args.force_download)
    if not ok:
        sys.exit(1)

    if args.setup:
        print("[OK] Setup concluído. SESCA pronto para uso.")
        sys.exit(0)

    if not check_sesca():
        sys.exit(1)

    if not args.pdb and not args.pdb_id:
        print("[ERRO] Forneça pelo menos um PDB com --pdb ou --pdb_id.")
        parser.print_help()
        sys.exit(1)

    run_pipeline(
        pdb_files=list(args.pdb),
        pdb_ids=args.pdb_id,
        output_dir=args.output,
        basis=args.basis,
        plot=args.plot,
        clean=args.clean,
    )


if __name__ == "__main__":
    main()
