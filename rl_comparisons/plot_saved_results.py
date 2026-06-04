import json
import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.analysis import plot_rl_comparison
    
def generate_plots_from_json():
    json_path = "../results/master_results.json"
    
    if not os.path.exists(json_path):
        print(f"⚠️ Ficheiro {json_path} não encontrado!")
        print("Tens de correr o 'python compare_rl.py' pelo menos uma vez para treinar e guardar os dados.")
        return

    print("A carregar os dados das simulações passadas...")
    with open(json_path, "r", encoding="utf-8") as f:
        master_results = json.load(f)

    print("A gerar os gráficos...")
    plot_rl_comparison(master_results, filename="../results/rl_comparison.png")
    print("✅ Gráficos atualizados com sucesso!")

if __name__ == "__main__":
    generate_plots_from_json()