"""Visualización de resultados finales del experimento.

Este módulo genera una infografía a partir del CSV de test más reciente para
resumir el comportamiento del modelo frente a la persistencia.
"""

import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class ResultsPlotter:
    """Genera gráficos de resumen a partir de los resultados de test."""

    def plot_latest_results(self):
        """Localiza el CSV más reciente y construye la figura resumen."""

        list_of_files = glob.glob("docs/resultados/*.csv")
        if not list_of_files:
            print("No se encontraron archivos CSV en docs/resultados")
            return

        latest_file = max(list_of_files, key=os.path.getctime)
        print(f"Graficando resultados de: {latest_file}")
        df = pd.read_csv(latest_file)

        step_means = df.groupby("step").agg(
            {
                "rmse_aurora": "mean",
                "mae_aurora": "mean",
                "rmse_persist": "mean",
            }
        ).reset_index()

        step_means["skill"] = (
            (step_means["rmse_persist"] - step_means["rmse_aurora"])
            / step_means["rmse_persist"]
            * 100
        )

        steps = step_means["step"].values
        labels = [f"{int(s)}h" for s in steps]

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 7))

        ax1.plot(
            steps,
            step_means["rmse_aurora"],
            "o-",
            color="dodgerblue",
            label="RMSE Aurora",
            linewidth=3,
            markersize=8,
        )
        ax1.plot(
            steps,
            step_means["rmse_persist"],
            "x--",
            color="crimson",
            label="RMSE Persistencia",
            linewidth=2,
            alpha=0.8,
        )

        ax1.set_title("Comparativa de Error (Unidades Reales)", fontsize=14, fontweight="bold", pad=15)
        ax1.set_xlabel("Horizonte Temporal")
        ax1.set_ylabel("Error (m/s)")
        ax1.set_xticks(steps)
        ax1.set_xticklabels(labels)
        ax1.legend(frameon=True, shadow=True)

        sns.boxplot(x="step", y="mae_aurora", data=df, ax=ax2, palette="Blues", showfliers=False)
        ax2.set_title("DispersiÃ³n del MAE por Paso", fontsize=14, fontweight="bold", pad=15)
        ax2.set_xlabel("Horizonte Temporal")
        ax2.set_ylabel("Error Absoluto (m/s)")
        ax2.set_xticklabels(labels)

        colors = ["#2ecc71" if value >= 0 else "#e74c3c" for value in step_means["skill"]]
        bars = ax3.bar(steps, step_means["skill"], color=colors, alpha=0.8, edgecolor="black", linewidth=1)

        ax3.set_title("Skill Score vs Persistencia", fontsize=14, fontweight="bold", pad=15)
        ax3.set_ylabel("Mejora respecto a Persistencia (%)")
        ax3.set_xlabel("Horizonte Temporal")
        ax3.set_xticks(steps)
        ax3.set_xticklabels(labels)
        ax3.axhline(0, color="black", linewidth=1.5)

        for bar in bars:
            height = bar.get_height()
            va = "bottom" if height > 0 else "top"
            offset = 1 if height > 0 else -4
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + offset,
                f"{height:.1f}%",
                ha="center",
                va=va,
                fontweight="bold",
                fontsize=10,
            )

        y_min = min(step_means["skill"].min() - 10, -10)
        y_max = max(step_means["skill"].max() + 10, 10)
        ax3.set_ylim(y_min, y_max)

        output_img = latest_file.replace(".csv", "_analisis_tfg.png")
        plt.tight_layout()
        plt.savefig(output_img, dpi=300)
        print(f"InfografÃ­a generada con Ã©xito: {output_img}")
        plt.show()


def plot_latest_results():
    """Mantiene la API histórica basada en función para el graficado final."""

    return ResultsPlotter().plot_latest_results()


if __name__ == "__main__":
    plot_latest_results()
