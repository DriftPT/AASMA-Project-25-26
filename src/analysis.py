import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def interpret_state(state):
    """
    Traduz o tuplo de estado para uma frase em português percetível.
    Assumimos o formato: (health, safe_distance, zombie_distance, safe_zone_discovered, closest_injured, team_too_far)
    """
    hp, dist_zom, count_zom, safe_phase, inj_near, team_far = state
    
    desc = []
    
    # 1. Agent Health
    if hp == 'low': desc.append("Critical HP")
    elif hp == 'medium': desc.append("Medium HP")
    elif hp == 'high': desc.append("High HP")
    
    # 2 and 3. Zombie Threat (Distance and Quantity)
    if dist_zom == 'danger': 
        desc.append(f"Zombies very close ({count_zom})")
    elif dist_zom == 'near': 
        desc.append(f"Zombies nearby ({count_zom})")
    elif dist_zom == 'far': 
        desc.append(f"Zombies far away")
    
    # 4. Safe Zone Knowledge
    if safe_phase == 'unknown': desc.append("SafeZone Unknown")
    elif safe_phase == 'near': desc.append("SafeZone Nearby")
    elif safe_phase == 'medium' or safe_phase == 'far': desc.append("SafeZone Far")
    
    # 5. Teammates Status
    if inj_near == 1: desc.append("Injured Teammate")
    
    # 6. Distance to Team
    if team_far == 1: desc.append("Far from Team")
        
    return " | ".join(desc)

def calculate_confidence(q_values_dict):
    """Calcula o grau de decisão do agente (Diferença entre a melhor ação e a média)"""
    values = list(q_values_dict.values())
    max_val = max(values)
    mean_val = sum(values) / len(values)
    return max_val - mean_val

def extract_top_states(q_table, top_n=15):
    """Extrai e ordena os estados onde o agente tem mais 'certeza' do que fazer."""
    state_metrics = []
    
    for state, actions in q_table.items():
        # Ignora estados onde o agente ainda não aprendeu nada (tudo a 0)
        if all(v == 0 for v in actions.values()):
            continue
            
        confidence = calculate_confidence(actions)
        best_action = max(actions, key=actions.get)
        
        state_metrics.append({
            "state_tuple": state,
            "q_values": actions,
            "confidence": confidence,
            "best_action": best_action
        })
        
    # Ordena pelos que têm maior confiança
    state_metrics.sort(key=lambda x: x["confidence"], reverse=True)
    return state_metrics[:top_n]


def plot_qtable_heatmap(q_table, filename="results/qtable_heatmap.png"):
    """Gera um Heatmap dos 15 estados mais visitados/decididos."""
    top_15 = extract_top_states(q_table, top_n=15)
    
    if not top_15:
        print("A Q-table não tem dados suficientes para o Heatmap.")
        return

    # Prepara dados para o Pandas
    index_labels = [interpret_state(item["state_tuple"]) for item in top_15]
    data = [item["q_values"] for item in top_15]
    
    df = pd.DataFrame(data, index=index_labels)
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=.5)
    plt.title("Q-Values for the 15 States with Highest Confidence")
    plt.ylabel("Observed State")
    plt.xlabel("Actions")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"✅ Guardado: {filename}")


def plot_learning_curve(training_history, baseline_metrics=None, out_dir="results"):
    """Gera gráficos de aprendizagem: um geral e um para cada modo de equipa (sem steps)."""
    df = pd.DataFrame(training_history)
    
    if df.empty:
        return

    # Usamos rolling mean para suavizar as linhas
    window = 50
    
    # ==========================================
    # 1. GRÁFICO PRINCIPAL (Média de tudo)
    # ==========================================
    df_geral = df.copy()
    df_geral['smooth_success'] = df_geral['success'].rolling(window, min_periods=1).mean()
    df_geral['smooth_survivors'] = df_geral['survivors'].rolling(window, min_periods=1).mean()
    
    # Agora só temos 2 subplots em vez de 3
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Success Rate
    sns.lineplot(data=df_geral, x='episode', y='smooth_success', ax=axes[0], color='blue')
    axes[0].set_title('Success Rate Over Training (Overall)')
    axes[0].set_ylabel('Success (0 to 1)')
    if baseline_metrics and 'success_rate' in baseline_metrics:
        axes[0].axhline(y=baseline_metrics['success_rate'], color='red', linestyle='--', label='Baseline')
        axes[0].legend()

    # Average Survivors
    sns.lineplot(data=df_geral, x='episode', y='smooth_survivors', ax=axes[1], color='green')
    axes[1].set_title('Average Survivors (Overall)')
    axes[1].set_ylabel('Survivors (0 to 3)')
    axes[1].set_xlabel('Training Episode')
    if baseline_metrics and 'avg_survivors' in baseline_metrics:
        axes[1].axhline(y=baseline_metrics['avg_survivors'], color='red', linestyle='--', label='Baseline')
        axes[1].legend()

    plt.tight_layout()
    filename_geral = f"{out_dir}/learning_curve_geral.png"
    plt.savefig(filename_geral, dpi=300)
    plt.close()
    print(f"✅ Guardado: {filename_geral}")

    # ==========================================
    # 2. GRÁFICOS INDIVIDUAIS POR MODO DE EQUIPA
    # ==========================================
    team_modes = df['team_mode'].unique()
    
    for mode in team_modes:
        # Filtramos os dados apenas para este modo específico
        df_mode = df[df['team_mode'] == mode].copy()
        
        # Recalculamos a média móvel SÓ para os episódios deste modo
        df_mode['smooth_success'] = df_mode['success'].rolling(window, min_periods=1).mean()
        df_mode['smooth_survivors'] = df_mode['survivors'].rolling(window, min_periods=1).mean()
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        # Success Rate
        sns.lineplot(data=df_mode, x='episode', y='smooth_success', ax=axes[0], color='blue')
        axes[0].set_title(f'Success Rate - {mode}')
        axes[0].set_ylabel('Success (0 to 1)')
        if baseline_metrics and 'success_rate' in baseline_metrics:
            axes[0].axhline(y=baseline_metrics['success_rate'], color='red', linestyle='--', label='Baseline')
            axes[0].legend()

        # Average Survivors
        sns.lineplot(data=df_mode, x='episode', y='smooth_survivors', ax=axes[1], color='green')
        axes[1].set_title(f'Average Survivors - {mode}')
        axes[1].set_ylabel('Survivors (0 to 3)')
        axes[1].set_xlabel('Training Episode')
        if baseline_metrics and 'avg_survivors' in baseline_metrics:
            axes[1].axhline(y=baseline_metrics['avg_survivors'], color='red', linestyle='--', label='Baseline')
            axes[1].legend()

        plt.tight_layout()
        
        # Formatar o nome do ficheiro para não ter espaços ou caracteres estranhos
        clean_mode_name = mode.replace(' ', '_').replace('(', '').replace(')', '').lower()
        filename_mode = f"{out_dir}/learning_curve_{clean_mode_name}.png"
        
        plt.savefig(filename_mode, dpi=300)
        plt.close()
        print(f"✅ Guardado: {filename_mode}")

def plot_test_comparisons(all_results, filename="results/test_comparison.png"):
    """
    Gera gráficos de barras comparando os resultados finais de teste 
    entre a Baseline e as diferentes configurações do agente Adaptativo.
    """
    if not all_results:
        return

    # Preparar os dados para o Pandas
    data = []
    for experiment_name, metrics in all_results.items():
        # Simplifica o nome para caber melhor no eixo Y do gráfico
        short_name = experiment_name.replace("Baseline: Scout + Defender + Support", "Baseline")
        short_name = short_name.replace("Adaptive replaces ", "Adapts ")
        short_name = short_name.replace("Adaptive (All Roles)", "Adapts (All)")

        data.append({
            "Teams": short_name,
            "Success Rate": metrics.get("success_rate", 0),
            "Average Survivors": metrics.get("avg_survivors", 0)
        })
        
    df = pd.DataFrame(data)

    # Configurar a imagem com 2 subplots lado a lado
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gráfico 1: Taxa de Sucesso
    sns.barplot(data=df, x="Success Rate", y="Teams", hue="Teams", ax=axes[0], palette="Blues_d", legend=False)
    axes[0].set_title('Success Rate Comparison in Final Test', fontweight='bold')
    axes[0].set_xlim(0, 1.0)
    # Adicionar os valores nas barras
    for i, v in enumerate(df["Success Rate"]):
        axes[0].text(v + 0.01, i, f"{v:.2f}", color='black', va='center')

    # Gráfico 2: Média de Sobreviventes
    sns.barplot(data=df, x="Average Survivors", y="Teams", hue="Teams", ax=axes[1], palette="Greens_d", legend=False)
    axes[1].set_title('Survivor Comparison in Final Test', fontweight='bold')
    axes[1].set_xlim(0, 3.0)
    # Adicionar os valores nas barras
    for i, v in enumerate(df["Average Survivors"]):
        axes[1].text(v + 0.03, i, f"{v:.2f}", color='black', va='center')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"✅ Guardado: {filename}")

def plot_role_ratios(all_results, filename="results/role_ratios.png"):
    """
    Gera um gráfico de barras empilhadas com a percentagem de uso 
    de cada papel pelo agente Adaptativo.
    """
    data = []
    for experiment_name, metrics in all_results.items():
        if "Baseline" in experiment_name:
            continue # A Baseline não tem agente adaptativo, logo ignoramos

        # Encurtar o nome para caber bem no eixo X
        short_name = experiment_name.replace("Adaptive replaces ", "Adapts ").replace("Adaptive (All Roles)", "Adapts (All)")
        
        # Vamos buscar as métricas (assumindo que as tens, com fallback para 0)
        scout_ratio = metrics.get("adaptive_scout_ratio", 0)
        defender_ratio = metrics.get("adaptive_defender_ratio", 0)
        support_ratio = metrics.get("adaptive_support_ratio", 0)
        
        # Normalizar para garantir que a soma é 100% (caso forneças os valores como contagens absolutas)
        total = scout_ratio + defender_ratio + support_ratio
        if total > 0:
            scout_ratio = (scout_ratio / total) * 100
            defender_ratio = (defender_ratio / total) * 100
            support_ratio = (support_ratio / total) * 100
            
        data.append({
            "Teams": short_name,
            "Scout": scout_ratio,
            "Defender": defender_ratio,
            "Support": support_ratio
        })
        
    if not data:
        print("Não há dados de rácios para gerar o gráfico de barras empilhadas.")
        return
        
    # Usar o pandas para facilitar o gráfico empilhado
    df = pd.DataFrame(data).set_index("Teams")
    
    # Gerar o gráfico com cores distintas
    ax = df.plot(kind='bar', stacked=True, figsize=(10, 6), color=['#4c72b0', '#55a868', '#c44e52'], edgecolor='white')
    
    plt.title('Role Distribution Chosen by the Adaptive Agent', fontweight='bold', pad=15)
    plt.ylabel('Usage Percentage (%)')
    plt.xlabel('Adaptive Scenarios')
    
    # Colocar a legenda fora do gráfico para não tapar as barras
    plt.legend(title="Assumed Role", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Rodar os labels do eixo X para se lerem melhor
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"✅ Guardado: {filename}")


def plot_events_table(all_results, filename="results/events_table.png"):
    """
    Gera uma tabela com as métricas médias de eventos por episódio.
    """
    if not all_results:
        return

    data = []
    for experiment_name, metrics in all_results.items():
        short_name = experiment_name.replace("Baseline: Scout + Defender + Support", "Baseline").replace("Adaptive replaces ", "Adapts ").replace("Adaptive (All Roles)", "Adapts (All)")
        
        # Extrair as métricas formatadas com 1 casa decimal (ou 2 para os scans que costumam ser < 1)
        data.append([
            short_name,
            f"{metrics.get('avg_steps', 0):.1f}",
            f"{metrics.get('avg_attack_events', 0):.1f}",
            f"{metrics.get('avg_heal_events', 0):.1f}",
            f"{metrics.get('avg_scan_events', 0):.2f}"
        ])
        
    columns = ["Teams", "Avg Steps", "Avg Attacks", "Avg Heals", "Avg Scans"]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.2) # Ajusta o tamanho das células
    
    # Estilizar o cabeçalho (linha 0) para ficar a azul escuro
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4c72b0')
            
    plt.title("Summary of Average Events per Episode", pad=20, size=14, weight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"✅ Guardado: {filename}")

def plot_rl_comparison(master_results, filename="results/rl_comparison.png"):
    """
    Gera gráficos de barras comparando a Taxa de Sucesso e a Média de Sobreviventes
    para as diferentes configurações de RL, extraídas do compare_rl.py.
    """
    if not master_results:
        return

    data = []
    for config, experiments in master_results.items():
        for exp_name, metrics in experiments.items():
            if "Baseline" in exp_name: 
                continue # Ignoramos a baseline para focar apenas nos algoritmos RL
            
            # Simplificar nomes para não encavalar no gráfico
            short_exp = exp_name.replace("Adaptive replaces ", "Adapts ").replace("Adaptive (All Roles)", "Adapts (All)")
            short_cfg = config.replace("Q-Learning", "QL").replace("SARSA", "Sarsa").replace("decay=", "d=")

            data.append({
                "RL Configuration": short_cfg,
                "Scenario": short_exp,
                "Success Rate": metrics.get("success_rate", 0),
                "Average Survivors": metrics.get("avg_survivors", 0)
            })
    
    df = pd.DataFrame(data)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    # Gráfico 1: Taxa de Sucesso
    sns.barplot(data=df, x="RL Configuration", y="Success Rate", hue="Scenario", ax=axes[0], palette="Blues_d")
    axes[0].set_title("Success Rate Comparison by RL Configuration", fontweight="bold", fontsize=14)
    axes[0].set_ylim(0.65, 1.05)
    axes[0].tick_params(axis='x', rotation=15)
    axes[0].legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    
    # Gráfico 2: Média de Sobreviventes
    sns.barplot(data=df, x="RL Configuration", y="Average Survivors", hue="Scenario", ax=axes[1], palette="Greens_d")
    axes[1].set_title("Survivor Comparison by RL Configuration", fontweight="bold", fontsize=14)
    axes[1].set_ylim(1, 3.2)
    axes[1].tick_params(axis='x', rotation=15)
    axes[1].legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Guardado: {filename}")

