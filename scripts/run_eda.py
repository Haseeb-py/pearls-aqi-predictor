"""Generate reproducible EDA from local multi-city backfill artifacts."""

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pearls_aqi.features.store import load_training_data
from pearls_aqi.settings import settings


def _save(name: str, directory: Path) -> None:
    plt.tight_layout()
    plt.savefig(directory / name, dpi=150)
    plt.close()


def run_eda() -> None:
    data = load_training_data()
    output = settings.BASE_DIR / "artifacts" / "eda"
    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    data.isna().mean().sort_values(ascending=False).head(15).plot.barh(title="Missingness")
    _save("missingness.png", output)
    sns.boxplot(data=data, x="city_slug", y="aqi")
    plt.title("AQI distributions by city")
    _save("distributions.png", output)
    sns.lineplot(data=data, x="event_time_utc", y="aqi", hue="city_slug", errorbar=None)
    plt.title("Hourly AQI trends")
    _save("trends.png", output)
    seasonal = data.groupby(["city_slug", "month"], as_index=False)["aqi"].mean()
    sns.lineplot(data=seasonal, x="month", y="aqi", hue="city_slug", marker="o")
    plt.title("Monthly AQI seasonality")
    _save("seasonality.png", output)
    columns = [c for c in ["aqi", "pm2_5_ug_m3", "pm10_ug_m3", "ozone_ug_m3", "temperature_2m_c", "relative_humidity_2m_pct", "wind_speed_10m_kph"] if c in data]
    sns.heatmap(data[columns].corr(), cmap="coolwarm", center=0, annot=True, fmt=".2f")
    plt.title("AQI feature correlation")
    _save("correlation.png", output)
    q1, q3 = data["aqi"].quantile([0.25, 0.75])
    outliers = data[(data["aqi"] < q1 - 1.5 * (q3 - q1)) | (data["aqi"] > q3 + 1.5 * (q3 - q1))]
    summary = data.groupby("city_slug")["aqi"].agg(["count", "mean", "median", "min", "max"])
    summary["outliers"] = outliers.groupby("city_slug").size().reindex(summary.index, fill_value=0)
    summary.to_csv(output / "city_summary.csv")
    (output / "README.md").write_text(
        "# EDA summary\n\n```text\n" + summary.to_string() + "\n```\n",
        encoding="utf-8",
    )
    print(f"EDA artifacts written to {output}")


if __name__ == "__main__":
    run_eda()
