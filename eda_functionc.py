"""
Neste arquivo serão armazenadas as configurações e funções auxilirares para o desenvolvimento do EDA.

"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats as scipy_stats
from scipy.signal import spectrogram as scipy_spectrogram

#Configurações

FS = 42_000

SIGNAL_COLUMNS = [
    "Accelerometer 1 (m/s^2)",
    "Microphone (V)",
    "Accelerometer 2 (m/s^2)",
    "Accelerometer 3 (m/s^2)",
    "Temperature (Celsius)"
]

CLASS_NAMES = {
    "H_H": "Healthy",
    "R_U": "Unbalance",
    "R_M": "Misalignment",
    "B_R": "Bowed rotor"
}

SPEED_NAMES = {
    "1": "15 Hz",
    "3": "45 Hz"
}

LOAD_NAMES = {
    "0": "Unloaded",
    "1": "Loaded"
}

CLASS_ORDER = [
    "Healthy",
    "Unbalance",
    "Misalignment",
    "Bowed rotor"
]

CONDITION_ORDER = [
    "15 Hz - Unloaded",
    "15 Hz - Loaded",
    "45 Hz - Unloaded",
    "45 Hz - Loaded"
]

# Funções auxiliares
def parse_filename(file_path):
    stem = Path(file_path).stem
    parts = stem.split("_")

    primary, subtype, speed_code, load_code = parts
    class_code = f"{primary}_{subtype}"

    if class_code not in CLASS_NAMES:
        raise ValueError(f"Classe desconhecida: {class_code}")

    if speed_code not in SPEED_NAMES:
        raise ValueError(f"Código de velocidade desconhecido: {speed_code}")

    if load_code not in LOAD_NAMES:
        raise ValueError(f"Código de carga desconhecido: {load_code}")

    return {
        "file": Path(file_path),
        "filename": Path(file_path).name,
        "class_code": class_code,
        "class_name": CLASS_NAMES[class_code],
        "speed_code": speed_code,
        "speed": SPEED_NAMES[speed_code],
        "load_code": load_code,
        "load": LOAD_NAMES[load_code],
        "condition": f"{SPEED_NAMES[speed_code]} - {LOAD_NAMES[load_code]}"
    }


def build_metadata_table(data_dir):
    files = sorted(Path(data_dir).glob("*.csv"))
    metadata = pd.DataFrame([parse_filename(file) for file in files])
    metadata["class_name"] = pd.Categorical(metadata["class_name"], categories=CLASS_ORDER, ordered=True)
    metadata["condition"] = pd.Categorical(metadata["condition"], categories=CONDITION_ORDER, ordered=True)

    return metadata.sort_values(["class_name", "speed_code", "load_code"]).reset_index(drop=True)


def read_signal_file(file_path):
    df = pd.read_csv(file_path)

    if len(df.columns) == 5:
        current_columns = [str(col).strip() for col in df.columns]

        expected_fragments = [
            "Accelerometer",
            "Microphone",
            "Accelerometer",
            "Accelerometer",
            "Temperature"
        ]

        header_detected = all(
            fragment.lower() in column.lower()
            for fragment, column in zip(
                expected_fragments,
                current_columns
            )
        )

        if header_detected:
            df.columns = SIGNAL_COLUMNS

        else:
            df = pd.read_csv(
                file_path,
                header=None,
                names=SIGNAL_COLUMNS
            )

    else:
        df = pd.read_csv(
            file_path,
            header=None,
            names=SIGNAL_COLUMNS
        )

    for column in SIGNAL_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    return df


def plot_experimental_design(metadata):
    conditions = [("15 Hz", "Unloaded"), ("15 Hz", "Loaded"), ("45 Hz", "Unloaded"), ("45 Hz", "Loaded")]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    matrix = np.zeros((len(CLASS_ORDER), len(conditions)))
    for i, class_name in enumerate(CLASS_ORDER):
        for j, (speed, load) in enumerate(conditions):
            row = metadata[(metadata["class_name"].astype(str) == class_name) & (metadata["speed"] == speed) & (metadata["load"] == load)]
            if not row.empty:
                matrix[i, j] = 1
    ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap = "copper")
    for i, class_name in enumerate(CLASS_ORDER):
        for j, (speed, load) in enumerate(conditions):
            row = metadata[(metadata["class_name"].astype(str) == class_name) & (metadata["speed"] == speed) & (metadata["load"] == load)]
            if not row.empty:
                filename = row.iloc[0]["filename"].replace(".csv", "")
                ax.text(j, i, filename, ha="center", va="center", fontsize=10, fontweight="bold")
            else:
                ax.text(j, i, "Missing", ha="center", va="center", fontsize=9)
    ax.set_xticks(np.arange(len(CONDITION_ORDER)))
    ax.set_xticklabels(CONDITION_ORDER)
    ax.set_yticks(np.arange(len(CLASS_ORDER)))
    ax.set_yticklabels(CLASS_ORDER)
    ax.set_xlabel("Condição de Operação")
    ax.set_ylabel("Condição do Motor")
    ax.set_title("Composição do dataset por condição experimental")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_condition_comparison(metadata, condition, domain="time", signal_columns=None, window_seconds=0.2, max_freq=2000, n_harmonics=3):
    if condition not in CONDITION_ORDER:
        raise ValueError(f"Condição de operação desconhecida: {condition!r}. Opções: {CONDITION_ORDER}")

    if domain not in ("time", "frequency"):
        raise ValueError("domain deve ser 'time' ou 'frequency'")

    if signal_columns is None:
        signal_columns = [
            "Accelerometer 1 (m/s^2)",
            "Accelerometer 2 (m/s^2)",
            "Accelerometer 3 (m/s^2)",
            "Microphone (V)"
        ]  

    speed, load = condition.split(" - ")
    subset = metadata[(metadata["speed"] == speed) & (metadata["load"] == load)]
    nominal_freq = float(speed.split()[0])

    n_rows = len(signal_columns)
    n_cols = len(CLASS_ORDER)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2 * n_cols, 2.2 * n_rows),
        sharex=True,
        sharey="row"
    )
    axes = np.atleast_2d(axes)

    for col, class_name in enumerate(CLASS_ORDER):
        rows = subset[subset["class_name"].astype(str) == class_name]

        if rows.empty:
            for row in range(n_rows):
                ax = axes[row, col]
                ax.text(0.5, 0.5, "Sem dados", ha="center", va="center", fontsize=9)
                ax.set_axis_off()
            continue

        signal_df = read_signal_file(rows.iloc[0]["file"])

        for row, column_name in enumerate(signal_columns):
            ax = axes[row, col]
            values = signal_df[column_name].to_numpy()

            if domain == "time":
                n_samples = min(len(values), int(window_seconds * FS))
                t = np.arange(n_samples) / FS
                ax.plot(t, values[:n_samples], linewidth=0.6, color="tab:blue")
                if row == n_rows - 1:
                    ax.set_xlabel("Tempo (s)")
            else:
                n = len(values)
                freqs = np.fft.rfftfreq(n, d=1 / FS)
                magnitude = np.abs(np.fft.rfft(values - np.mean(values))) / n
                mask = freqs <= max_freq
                ax.plot(freqs[mask], magnitude[mask], linewidth=0.6, color="tab:blue")
                if row == n_rows - 1:
                    ax.set_xlabel("Frequência (Hz)")

            if row == 0:
                ax.set_title(class_name, fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(column_name, fontsize=8)

    domain_label = "Domínio do Tempo" if domain == "time" else "Domínio da Frequência"
    fig.suptitle(f"Comparação das condições do motor: {condition} ({domain_label})", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()


def plot_operating_condition_comparison(metadata, class_name, domain="time", signal_columns=None, window_seconds=0.2, max_freq=2000, n_harmonics=3):
    """
    Compara as quatro condições de operação (velocidade + carga) para uma única
    condição do motor (classe), no domínio do tempo ou da frequência.

    class_name: uma das strings em CLASS_ORDER, ex.: "Healthy".
    domain: "time" ou "frequency".
    """
    if class_name not in CLASS_ORDER:
        raise ValueError(f"Condição do motor desconhecida: {class_name!r}. Opções: {CLASS_ORDER}")

    if domain not in ("time", "frequency"):
        raise ValueError("domain deve ser 'time' ou 'frequency'")

    if signal_columns is None:
        signal_columns = [
            "Accelerometer 1 (m/s^2)",
            "Accelerometer 2 (m/s^2)",
            "Accelerometer 3 (m/s^2)",
            "Microphone (V)"
        ]  # canais vibro-acústicos, exclui Temperature; microfone por último

    subset = metadata[metadata["class_name"].astype(str) == class_name]

    n_rows = len(signal_columns)
    n_cols = len(CONDITION_ORDER)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2 * n_cols, 2.2 * n_rows),
        sharex=True,
        sharey="row"
    )
    axes = np.atleast_2d(axes)

    for col, condition in enumerate(CONDITION_ORDER):
        speed, load = condition.split(" - ")
        nominal_freq = float(speed.split()[0])
        rows = subset[(subset["speed"] == speed) & (subset["load"] == load)]

        if rows.empty:
            for row in range(n_rows):
                ax = axes[row, col]
                ax.text(0.5, 0.5, "Sem dados", ha="center", va="center", fontsize=9)
                ax.set_axis_off()
            continue

        signal_df = read_signal_file(rows.iloc[0]["file"])

        for row, column_name in enumerate(signal_columns):
            ax = axes[row, col]
            values = signal_df[column_name].to_numpy()

            if domain == "time":
                n_samples = min(len(values), int(window_seconds * FS))
                t = np.arange(n_samples) / FS
                ax.plot(t, values[:n_samples], linewidth=0.6, color="tab:blue")
                if row == n_rows - 1:
                    ax.set_xlabel("Tempo (s)")
            else:
                n = len(values)
                freqs = np.fft.rfftfreq(n, d=1 / FS)
                magnitude = np.abs(np.fft.rfft(values - np.mean(values))) / n
                mask = freqs <= max_freq
                ax.plot(freqs[mask], magnitude[mask], linewidth=0.6, color="tab:blue")
                for harmonic in range(1, n_harmonics + 1):
                    freq_line = nominal_freq * harmonic
                    if freq_line <= max_freq:
                        ax.axvline(freq_line, color="tab:red", linestyle="--", linewidth=0.6, alpha=0.6)
                if row == n_rows - 1:
                    ax.set_xlabel("Frequência (Hz)")

            if row == 0:
                ax.set_title(condition, fontsize=10, fontweight="bold")
            if col == 0:
                ax.set_ylabel(column_name, fontsize=8)

    domain_label = "Domínio do Tempo" if domain == "time" else "Domínio da Frequência"
    fig.suptitle(f"Comparação das condições de operação: {class_name} ({domain_label})", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()


def compute_signal_statistics(metadata, signal_columns=None):
    """
    Calcula estatísticas descritivas por arquivo e canal (uma linha por combinação
    arquivo x canal), como base para orientar a extração de features.
    """
    if signal_columns is None:
        signal_columns = SIGNAL_COLUMNS[:4]  # canais vibro-acústicos, exclui Temperature

    records = []
    for _, row in metadata.iterrows():
        signal_df = read_signal_file(row["file"])

        for column_name in signal_columns:
            values = signal_df[column_name].dropna().to_numpy()
            rms = np.sqrt(np.mean(values ** 2))
            peak = np.max(np.abs(values))

            records.append({
                "filename": row["filename"],
                "class_name": row["class_name"],
                "speed": row["speed"],
                "load": row["load"],
                "condition": row["condition"],
                "channel": column_name,
                "mean": np.mean(values),
                "std": np.std(values),
                "rms": rms,
                "min": np.min(values),
                "max": np.max(values),
                "peak_to_peak": np.ptp(values),
                "skewness": scipy_stats.skew(values),
                "kurtosis": scipy_stats.kurtosis(values),
                "crest_factor": peak / rms if rms > 0 else np.nan
            })

    stats_df = pd.DataFrame.from_records(records)
    stats_df["class_name"] = pd.Categorical(stats_df["class_name"], categories=CLASS_ORDER, ordered=True)
    stats_df["condition"] = pd.Categorical(stats_df["condition"], categories=CONDITION_ORDER, ordered=True)

    return stats_df


def plot_statistics_by_class(stats_df, metric="rms", signal_columns=None):
    """
    Compara a distribuição de uma métrica estatística (ex.: rms, kurtosis,
    crest_factor) entre as quatro condições do motor, um subplot por canal.
    """
    non_metric_columns = {"filename", "class_name", "speed", "load", "condition", "channel"}
    if metric not in stats_df.columns or metric in non_metric_columns:
        available = [c for c in stats_df.columns if c not in non_metric_columns]
        raise ValueError(f"Métrica desconhecida: {metric!r}. Opções: {available}")

    if signal_columns is None:
        signal_columns = [
            "Accelerometer 1 (m/s^2)",
            "Accelerometer 2 (m/s^2)",
            "Accelerometer 3 (m/s^2)",
            "Microphone (V)"
        ]  

    fig, axes = plt.subplots(1, len(signal_columns), figsize=(4 * len(signal_columns), 4.2))
    axes = np.atleast_1d(axes)

    for ax, channel in zip(axes, signal_columns):
        channel_data = stats_df[stats_df["channel"] == channel]
        data_by_class = [
            channel_data[channel_data["class_name"].astype(str) == class_name][metric].to_numpy()
            for class_name in CLASS_ORDER
        ]
        ax.boxplot(data_by_class, tick_labels=CLASS_ORDER, showmeans=True)
        for i, values in enumerate(data_by_class, start=1):
            jitter = np.random.normal(0, 0.04, size=len(values))
            ax.scatter(np.full(len(values), i) + jitter, values, color="tab:blue", s=15, alpha=0.7, zorder=3)
        ax.set_title(channel, fontsize=10)
        ax.tick_params(axis="x", rotation=30)

    axes[0].set_ylabel(metric)
    fig.suptitle(f"Distribuição de '{metric}' por condição do motor", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.show()


def check_signal_quality(metadata, signal_columns=None):
    """
    Verifica qualidade dos dados por arquivo e canal: valores ausentes (NaN) e
    indício de saturação/clipping (proporção de amostras exatamente no valor
    mínimo ou máximo observado, já que um sinal contínuo saturado tende a "grudar"
    no valor de rail, diferente de um pico suave e isolado).
    """
    if signal_columns is None:
        signal_columns = SIGNAL_COLUMNS  

    records = []
    for _, row in metadata.iterrows():
        signal_df = read_signal_file(row["file"])

        for column_name in signal_columns:
            series = signal_df[column_name]
            n_total = len(series)
            n_missing = int(series.isna().sum())

            values = series.dropna().to_numpy()
            if len(values) > 0:
                at_max = np.isclose(values, np.max(values))
                at_min = np.isclose(values, np.min(values))
                pct_at_extreme = 100 * (at_max.sum() + at_min.sum()) / len(values)
            else:
                pct_at_extreme = np.nan

            records.append({
                "filename": row["filename"],
                "class_name": row["class_name"],
                "condition": row["condition"],
                "channel": column_name,
                "n_missing": n_missing,
                "pct_missing": 100 * n_missing / n_total,
                "pct_at_extreme_value": pct_at_extreme
            })

    return pd.DataFrame.from_records(records)


def plot_signal_trend(metadata, class_name, condition, signal_columns=None, window_seconds=0.1):
    """
    Mostra a tendência do RMS ao longo de toda a gravação (10s), calculado em
    janelas curtas, para verificar estacionariedade e possíveis transientes
    (ex.: ramp-up do motor no início da gravação).
    """
    if class_name not in CLASS_ORDER:
        raise ValueError(f"Condição do motor desconhecida: {class_name!r}. Opções: {CLASS_ORDER}")

    if condition not in CONDITION_ORDER:
        raise ValueError(f"Condição de operação desconhecida: {condition!r}. Opções: {CONDITION_ORDER}")

    if signal_columns is None:
        signal_columns = [
            "Accelerometer 1 (m/s^2)",
            "Accelerometer 2 (m/s^2)",
            "Accelerometer 3 (m/s^2)",
            "Microphone (V)"
        ]  

    speed, load = condition.split(" - ")
    rows = metadata[(metadata["class_name"].astype(str) == class_name) & (metadata["speed"] == speed) & (metadata["load"] == load)]

    if rows.empty:
        raise ValueError(f"Nenhum arquivo encontrado para {class_name!r} em {condition!r}.")

    signal_df = read_signal_file(rows.iloc[0]["file"])
    window_size = int(window_seconds * FS)
    n_windows = len(signal_df) // window_size

    fig, axes = plt.subplots(len(signal_columns), 1, figsize=(10, 2.0 * len(signal_columns)), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, column_name in zip(axes, signal_columns):
        values = signal_df[column_name].to_numpy()[:n_windows * window_size].reshape(n_windows, window_size)
        rms_trend = np.sqrt(np.mean(values ** 2, axis=1))
        t = (np.arange(n_windows) + 0.5) * window_seconds
        ax.plot(t, rms_trend, color="tab:blue", linewidth=1.0)
        ax.set_ylabel(column_name, fontsize=8)

    axes[-1].set_xlabel("Tempo (s)")
    fig.suptitle(f"Tendência do RMS ao longo da gravação: {class_name} | {condition}", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()


