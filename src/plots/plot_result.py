"""Visualizacion de resultados finales del experimento.

Este modulo genera una infografia a partir del CSV de test mas reciente para
resumir el comportamiento del modelo frente a varios baselines.
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class ResultsPlotter:
    """Genera graficos de resumen a partir de los resultados de test."""

    def plot_latest_results(self):
        """Localiza el CSV mas reciente y construye la figura resumen."""

        list_of_files = glob.glob("docs/resultados/*.csv")
        if not list_of_files:
            print("No se encontraron archivos CSV en docs/resultados")
            return

        latest_file = max(list_of_files, key=os.path.getctime)
        print(f"Graficando resultados de: {latest_file}")
        df = pd.read_csv(latest_file)

        step_means = df.groupby("step").agg(
            {
                "rmse_aurora_ft": "mean",
                "rmse_aurora_base": "mean",
                "mae_aurora_ft": "mean",
                "rmse_persist": "mean",
                "rmse_clima": "mean",
            }
        ).reset_index()

        step_means["skill_vs_persist"] = (
            (step_means["rmse_persist"] - step_means["rmse_aurora_ft"])
            / step_means["rmse_persist"]
            * 100
        )
        step_means["skill_vs_clima"] = (
            (step_means["rmse_clima"] - step_means["rmse_aurora_ft"])
            / step_means["rmse_clima"]
            * 100
        )
        step_means["gain_vs_base"] = (
            (step_means["rmse_aurora_base"] - step_means["rmse_aurora_ft"])
            / step_means["rmse_aurora_base"]
            * 100
        )

        steps = step_means["step"].values
        labels = [f"{int(s)}h" for s in steps]

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))

        ax1.plot(
            steps,
            step_means["rmse_aurora_ft"],
            "o-",
            color="dodgerblue",
            label="Aurora Fine-tuned",
            linewidth=3,
            markersize=8,
        )
        ax1.plot(
            steps,
            step_means["rmse_aurora_base"],
            "o--",
            color="#8ec5ff",
            label="Aurora Base",
            linewidth=2.5,
            markersize=7,
        )
        ax1.plot(
            steps,
            step_means["rmse_persist"],
            "x-",
            color="crimson",
            label="Persistencia",
            linewidth=2,
            alpha=0.9,
        )
        ax1.plot(
            steps,
            step_means["rmse_clima"],
            "s-",
            color="gray",
            label="Climatologia",
            linewidth=2,
            alpha=0.85,
        )

        ax1.set_title("Comparativa de RMSE", fontsize=14, fontweight="bold", pad=15)
        ax1.set_xlabel("Horizonte Temporal")
        ax1.set_ylabel("Error (m/s)")
        ax1.set_xticks(steps)
        ax1.set_xticklabels(labels)
        ax1.legend(frameon=True, shadow=True)

        sns.boxplot(x="step", y="mae_aurora_ft", data=df, ax=ax2, palette="Blues", showfliers=False)
        ax2.set_title("Dispersion del MAE del Fine-tuning", fontsize=14, fontweight="bold", pad=15)
        ax2.set_xlabel("Horizonte Temporal")
        ax2.set_ylabel("Error Absoluto (m/s)")
        ax2.set_xticklabels(labels)

        x = np.arange(len(steps))
        width = 0.24
        bars_persist = ax3.bar(
            x - width,
            step_means["skill_vs_persist"],
            width=width,
            color="crimson",
            alpha=0.8,
            edgecolor="black",
            linewidth=1,
            label="vs Persistencia",
        )
        bars_clima = ax3.bar(
            x,
            step_means["skill_vs_clima"],
            width=width,
            color="gray",
            alpha=0.8,
            edgecolor="black",
            linewidth=1,
            label="vs Climatologia",
        )
        bars_base = ax3.bar(
            x + width,
            step_means["gain_vs_base"],
            width=width,
            color="dodgerblue",
            alpha=0.85,
            edgecolor="black",
            linewidth=1,
            label="Ganancia vs Base",
        )

        ax3.set_title("Skill Scores del Fine-tuning", fontsize=14, fontweight="bold", pad=15)
        ax3.set_ylabel("Mejora (%)")
        ax3.set_xlabel("Horizonte Temporal")
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels)
        ax3.axhline(0, color="black", linewidth=1.5)
        ax3.legend(frameon=True)

        for bars in (bars_persist, bars_clima, bars_base):
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
                    fontsize=9,
                )

        y_min = min(
            step_means["skill_vs_persist"].min(),
            step_means["skill_vs_clima"].min(),
            step_means["gain_vs_base"].min(),
        ) - 10
        y_max = max(
            step_means["skill_vs_persist"].max(),
            step_means["skill_vs_clima"].max(),
            step_means["gain_vs_base"].max(),
        ) + 10
        ax3.set_ylim(y_min, y_max)

        output_img = latest_file.replace(".csv", "_analisis_tfg.png")
        plt.tight_layout()
        plt.savefig(output_img, dpi=300)
        print(f"Infografia generada con exito: {output_img}")
        plt.show()


def plot_latest_results():
    """Mantiene la API historica basada en funcion para el graficado final."""

    return ResultsPlotter().plot_latest_results()


if __name__ == "__main__":
    plot_latest_results()
