import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

def plot_latest_results():
    # 1. Buscar el último CSV generado en docs/resultados
    list_of_files = glob.glob('docs/resultados/*.csv')
    if not list_of_files:
        print("No se encontraron archivos CSV en docs/resultados")
        return
    
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Graficando resultados de: {latest_file}")
    
    df = pd.read_csv(latest_file)
    
    # Configuración de estilo
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- GRÁFICO 1: Histograma de Distribución del Error ---
    sns.histplot(df['rmse_aurora'], kde=True, color='skyblue', label='Aurora (Fine-tuned)', ax=ax1)
    sns.histplot(df['rmse_persist'], kde=True, color='salmon', label='Persistencia', ax=ax1)
    ax1.set_title('Distribución del Error RMSE (Test)', fontsize=14)
    ax1.set_xlabel('RMSE')
    ax1.set_ylabel('Frecuencia')
    ax1.legend()

    # --- GRÁFICO 2: Comparativa de Medias ---
    means = [df['rmse_aurora'].mean(), df['rmse_persist'].mean()]
    labels = ['Aurora', 'Persistencia']
    colors = ['skyblue', 'salmon']
    
    bars = ax2.bar(labels, means, color=colors, alpha=0.8)
    ax2.set_title('Comparativa de RMSE Medio', fontsize=14)
    ax2.set_ylabel('RMSE Medio')
    
    # Añadir el valor encima de las barras
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Guardar la imagen para la memoria
    output_img = latest_file.replace(".csv", ".png")
    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    print(f"Gráfica guardada en: {output_img}")
    plt.show()

if __name__ == "__main__":
    plot_latest_results()