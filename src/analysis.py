import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Garante que a pasta para as imagens existe
if not os.path.exists('results'):
    os.makedirs('results')

def interpret_state(state):
    """
    Traduz o tuplo de estado para uma frase em português percetível.
    Assumimos o formato: (health, safe_distance, zombie_distance, safe_zone_discovered, closest_injured, team_too_far)
    """
    hp, dist_zom, count_zom, safe_phase, inj_near, team_far = state
    
    desc = []
    
    # 1. Vida do Agente
    if hp == 'low': desc.append("HP Crítico")
    elif hp == 'medium': desc.append("HP Médio")
    elif hp == 'high': desc.append("HP Alto")
    
    # 2 e 3. Ameaça Zombie (Distância e Quantidade)
    if dist_zom == 'danger': 
        desc.append(f"Zumbis colados ({count_zom})")
    elif dist_zom == 'near': 
        desc.append(f"Zumbis perto ({count_zom})")
    elif dist_zom == 'far': 
        desc.append(f"Zumbis longe")
    
    # 4. Conhecimento da Base Segura
    if safe_phase == 'unknown': desc.append("SafeZone Desconhecida")
    elif safe_phase == 'near': desc.append("SafeZone Perto")
    elif safe_phase == 'medium' or safe_phase == 'far': desc.append("SafeZone Longe")
    
    # 5. Estado dos Colegas
    if inj_near == 1: desc.append("Colega Ferido")
    
    # 6. Distância da Equipa
    if team_far == 1: desc.append("Longe da Equipa")
        
    return " | ".join(desc)

def calculate_confidence(q_values_dict):
    """Calcula o grau de decisão do agente (Diferença entre a melhor ação e a média)"""
    values = list(q_values_dict.values())
    max_val = max(values)
    mean_val = sum(values) / len(values)
    return max_val - mean_val

def extract_top_states(q_table, top_n=30):
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
    """Gera um Heatmap dos 30 estados mais visitados/decididos."""
    top_30 = extract_top_states(q_table, top_n=30)
    
    if not top_30:
        print("A Q-table não tem dados suficientes para o Heatmap.")
        return

    # Prepara dados para o Pandas
    index_labels = [interpret_state(item["state_tuple"]) for item in top_30]
    data = [item["q_values"] for item in top_30]
    
    df = pd.DataFrame(data, index=index_labels)
    
    plt.figure(figsize=(10, 12))
    sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=.5)
    plt.title("Q-Values para os 30 Estados com Maior Confiança")
    plt.ylabel("Estado Ocorrido")
    plt.xlabel("Ações")
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
    
    # Taxa de Sucesso
    sns.lineplot(data=df_geral, x='episode', y='smooth_success', ax=axes[0], color='blue')
    axes[0].set_title('Taxa de Sucesso ao Longo do Treino (Geral)')
    axes[0].set_ylabel('Sucesso (0 a 1)')
    if baseline_metrics and 'success_rate' in baseline_metrics:
        axes[0].axhline(y=baseline_metrics['success_rate'], color='red', linestyle='--', label='Baseline')
        axes[0].legend()

    # Média de Sobreviventes
    sns.lineplot(data=df_geral, x='episode', y='smooth_survivors', ax=axes[1], color='green')
    axes[1].set_title('Média de Sobreviventes (Geral)')
    axes[1].set_ylabel('Sobreviventes (0 a 3)')
    axes[1].set_xlabel('Episódio de Treino')
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
        
        # Taxa de Sucesso
        sns.lineplot(data=df_mode, x='episode', y='smooth_success', ax=axes[0], color='blue')
        axes[0].set_title(f'Taxa de Sucesso - {mode}')
        axes[0].set_ylabel('Sucesso (0 a 1)')
        if baseline_metrics and 'success_rate' in baseline_metrics:
            axes[0].axhline(y=baseline_metrics['success_rate'], color='red', linestyle='--', label='Baseline')
            axes[0].legend()

        # Média de Sobreviventes
        sns.lineplot(data=df_mode, x='episode', y='smooth_survivors', ax=axes[1], color='green')
        axes[1].set_title(f'Média de Sobreviventes - {mode}')
        axes[1].set_ylabel('Sobreviventes (0 a 3)')
        axes[1].set_xlabel('Episódio de Treino')
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
            "Equipas": short_name,
            "Taxa de Sucesso": metrics.get("success_rate", 0),
            "Média Sobreviventes": metrics.get("avg_survivors", 0)
        })
        
    df = pd.DataFrame(data)

    # Configurar a imagem com 2 subplots lado a lado
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gráfico 1: Taxa de Sucesso
    sns.barplot(data=df, x="Taxa de Sucesso", y="Equipas", hue="Equipas", ax=axes[0], palette="Blues_d", legend=False)
    axes[0].set_title('Comparação da Taxa de Sucesso no Teste Final', fontweight='bold')
    axes[0].set_xlim(0, 1.0)
    # Adicionar os valores nas barras
    for i, v in enumerate(df["Taxa de Sucesso"]):
        axes[0].text(v + 0.01, i, f"{v:.2f}", color='black', va='center')

    # Gráfico 2: Média de Sobreviventes
    sns.barplot(data=df, x="Média Sobreviventes", y="Equipas", hue="Equipas", ax=axes[1], palette="Greens_d", legend=False)
    axes[1].set_title('Comparação de Sobreviventes no Teste Final', fontweight='bold')
    axes[1].set_xlim(0, 3.0)
    # Adicionar os valores nas barras
    for i, v in enumerate(df["Média Sobreviventes"]):
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
            "Equipas": short_name,
            "Scout": scout_ratio,
            "Defender": defender_ratio,
            "Support": support_ratio
        })
        
    if not data:
        print("Não há dados de rácios para gerar o gráfico de barras empilhadas.")
        return
        
    # Usar o pandas para facilitar o gráfico empilhado
    df = pd.DataFrame(data).set_index("Equipas")
    
    # Gerar o gráfico com cores distintas
    ax = df.plot(kind='bar', stacked=True, figsize=(10, 6), color=['#4c72b0', '#55a868', '#c44e52'], edgecolor='white')
    
    plt.title('Distribuição de Papéis Escolhidos pelo Agente Adaptativo', fontweight='bold', pad=15)
    plt.ylabel('Percentagem de Uso (%)')
    plt.xlabel('Cenários com Adaptativo')
    
    # Colocar a legenda fora do gráfico para não tapar as barras
    plt.legend(title="Papel Assumido", bbox_to_anchor=(1.05, 1), loc='upper left')
    
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
        
    columns = ["Equipas", "Avg Steps", "Avg Attacks", "Avg Heals", "Avg Scans"]
    
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
            
    plt.title("Resumo de Eventos Médios por Episódio", pad=20, size=14, weight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"✅ Guardado: {filename}")