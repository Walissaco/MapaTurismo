# scripts/main.py
import subprocess
from pathlib import Path
import sys

# ====================================================================
# SCRIPT DE PREPARAÇÃO DE DADOS E TREINAMENTO DO MODELO
# (Projetado para ser chamado por outro script)
# ====================================================================

# --- 1. CONFIGURAÇÃO DE CAMINHOS ---
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
OUTPUT_NOTEBOOKS_DIR = ROOT_DIR / "notebooks_executed"
OUTPUT_NOTEBOOKS_DIR.mkdir(exist_ok=True)

NOTEBOOK_PIPELINE = [
    "data_preparation.ipynb",
    "model_training.ipynb",
    "map_visualization.ipynb"
]

# --- 2. FUNÇÃO PARA EXECUTAR NOTEBOOKS ---
def run_notebook(notebook_filename):
    input_path = NOTEBOOKS_DIR / notebook_filename
    output_path = OUTPUT_NOTEBOOKS_DIR / notebook_filename
    print(f"\n  Executando notebook: {notebook_filename}...")
    
    if not input_path.exists():
        print(f"    ERRO: Notebook '{input_path}' não encontrado. A pular.")
        return False

    try:
        subprocess.run(
            ["papermill", str(input_path), str(output_path), "--cwd", str(NOTEBOOKS_DIR)],
            check=True, capture_output=True, text=True, encoding='utf-8'
        )
        print(f"    Sucesso! Notebook executado salvo em: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"    ERRO ao executar '{notebook_filename}':")
        error_lines = e.stderr.strip().split('\n')
        print("      Últimas linhas do erro:", *error_lines[-5:], sep='\n         ')
        return False
    except FileNotFoundError:
        print("    ERRO: 'papermill' não encontrado. Instale com: pip install papermill")
        return False

# --- 3. EXECUÇÃO DO FLUXO PRINCIPAL ---
if __name__ == "__main__":
    print("=" * 50)
    print("      INICIANDO PIPELINE DE DADOS E TREINAMENTO     ")
    print("=" * 50)
    
    pipeline_success = all(run_notebook(notebook) for notebook in NOTEBOOK_PIPELINE)
            
    if pipeline_success:
        print("\n Pipeline de dados e treinamento concluído com sucesso!")
        # Sai com código 0 para indicar sucesso
        sys.exit(0)
    else:
        print("\n O pipeline de dados falhou.")
        # Sai com código 1 para indicar erro
        sys.exit(1)