import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

def plot_latest_results():
    list_of_files = glob.glob('docs/resultados/*.csv')
    if not list_of_files:
        print("No se encontraron archivos CSV en docs/resultados")
        return
    
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Graficando resultados de: {latest_file}")
    df = pd.read_csv(latest_file)
    
    
    step_means = df.groupby('step').mean().reset_index()
    
    step_means['skill'] = (step_means['rmse_persist'] - step_means['rmse_aurora']) / step_means['rmse_persist'] * 100

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    
    ax1.plot(step_means['step'], step_means['rmse_aurora'], 'o-', color='dodgerblue', label='RMSE Aurora', linewidth=2)
    ax1.plot(step_means['step'], step_means['mae_aurora'], 's--', color='forestgreen', label='MAE Aurora', linewidth=2)
    ax1.set_title('Evolución del Error (0-24h)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Paso de Pronóstico (6h cada uno)')
    ax1.set_ylabel('Error (m/s)')
    ax1.set_xticks([1, 2, 3, 4])
    ax1.set_xticklabels(['6h', '12h', '18h', '24h'])
    ax1.legend()

    
    sns.boxplot(x='step', y='mape_aurora', data=df, ax=ax2, palette='viridis', showfliers=False)
    ax2.set_title('Distribución del MAPE (%)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Horizonte Temporal')
    ax2.set_ylabel('MAPE (%)')
    ax2.set_xticklabels(['6h', '12h', '18h', '24h'])

    
    bars = ax3.bar(step_means['step'], step_means['skill'], color='orange', alpha=0.7)
    ax3.set_title('Skill Score vs Persistencia', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Mejora respecto a Persistencia (%)')
    ax3.set_xlabel('Horizonte Temporal')
    ax3.set_xticks([1, 2, 3, 4])
    ax3.set_xticklabels(['6h', '12h', '18h', '24h'])
    ax3.axhline(0, color='black', linewidth=1)

    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

    output_img = latest_file.replace(".csv", "_infografia.png")
    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    print(f"Infografía guardada en: {output_img}")
    plt.show()

if __name__ == "__main__":
    plot_latest_results()