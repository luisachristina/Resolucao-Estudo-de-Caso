"""
Neste arquivo serão armazenadas as configurações e funções auxilirares para a extração de features.

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_selection import f_classif, mutual_info_classif

from eda_functionc import FS, CLASS_ORDER, CONDITION_ORDER, read_signal_file

# Configurações

DEFAULT_SIGNAL_COLUMNS = [
    "Accelerometer 1 (m/s^2)",
    "Accelerometer 2 (m/s^2)",
    "Accelerometer 3 (m/s^2)",
    "Microphone (V)"
]  # canais vibro-acústicos, exclui Temperature; microfone por último

METADATA_COLUMNS = [
    "filename",
    "class_name",
    "speed",
    "load",
    "condition",
    "window_id",
    "start_time",
    "end_time"
]


# Extração de features por janela

def _time_domain_features(values):
    rms = np.sqrt(np.mean(values ** 2))
    peak = np.max(np.abs(values))

    return {
        "mean": np.mean(values),
        "std": np.std(values),
        "rms": rms,
        "peak_to_peak": np.ptp(values),
        "skewness": scipy_stats.skew(values),
        "kurtosis": scipy_stats.kurtosis(values),
        "crest_factor": peak / rms if rms > 0 else np.nan
    }


def _frequency_domain_features(values, nominal_freq, n_harmonics=3, max_freq=2000, harmonic_tolerance=0.05):
    n = len(values)
    freqs = np.fft.rfftfreq(n, d=1 / FS)
    magnitude = np.abs(np.fft.rfft(values - np.mean(values))) / n

    band_mask = freqs <= max_freq
    band_freqs = freqs[band_mask]
    band_magnitude = magnitude[band_mask]

    features = {}

    # Magnitude nos harmônicos da frequência de rotação, buscando o pico numa
    # pequena faixa ao redor de cada harmônico (tolerante ao escorregamento do
    # motor de indução, que desloca a frequência mecânica real em relação à
    # frequência nominal do VFD).
    for harmonic in range(1, n_harmonics + 1):
        center = nominal_freq * harmonic
        low, high = center * (1 - harmonic_tolerance), center * (1 + harmonic_tolerance)
        harmonic_mask = (band_freqs >= low) & (band_freqs <= high)
        features[f"harmonic_{harmonic}x_magnitude"] = band_magnitude[harmonic_mask].max() if harmonic_mask.any() else 0.0

    total_energy = np.sum(band_magnitude ** 2)
    features["spectral_energy"] = total_energy

    if total_energy > 0:
        features["spectral_centroid"] = np.sum(band_freqs * band_magnitude ** 2) / total_energy
        probs = (band_magnitude ** 2) / total_energy
        probs = probs[probs > 0]
        features["spectral_entropy"] = -np.sum(probs * np.log2(probs))
    else:
        features["spectral_centroid"] = 0.0
        features["spectral_entropy"] = 0.0

    dominant_idx = np.argmax(band_magnitude)
    features["dominant_frequency"] = band_freqs[dominant_idx]

    return features


def extract_features(metadata, window_seconds=1.0, overlap=0.5, signal_columns=None, n_harmonics=3, max_freq=2000):
    """
    Segmenta cada gravação em janelas de tamanho fixo (com sobreposição) e extrai,
    por janela e canal, features no domínio do tempo e da frequência.

    Retorna uma tabela com uma linha por janela: colunas de metadados
    (filename, class_name, speed, load, condition, window_id, start_time, end_time)
    + uma coluna por feature, nomeada como "{canal}__{feature}".

    A coluna "filename" identifica a gravação de origem de cada janela e deve ser
    usada como grupo em qualquer partição de validação cruzada (ver
    `nested_group_kfold_splits`), para que janelas da mesma gravação nunca fiquem
    em folds diferentes.
    """
    if not 0 <= overlap < 1:
        raise ValueError("overlap deve estar no intervalo [0, 1)")

    if signal_columns is None:
        signal_columns = DEFAULT_SIGNAL_COLUMNS

    window_size = int(window_seconds * FS)
    step = max(1, int(window_size * (1 - overlap)))

    records = []
    for _, row in metadata.iterrows():
        signal_df = read_signal_file(row["file"])
        nominal_freq = float(row["speed"].split()[0])
        n_samples_total = len(signal_df)

        channel_values = {column_name: signal_df[column_name].to_numpy() for column_name in signal_columns}

        for window_id, start in enumerate(range(0, n_samples_total - window_size + 1, step)):
            end = start + window_size

            record = {
                "filename": row["filename"],
                "class_name": row["class_name"],
                "speed": row["speed"],
                "load": row["load"],
                "condition": row["condition"],
                "window_id": window_id,
                "start_time": start / FS,
                "end_time": end / FS
            }

            for column_name in signal_columns:
                values = channel_values[column_name][start:end]
                channel_prefix = column_name.split(" (")[0].replace(" ", "_").lower()

                time_features = _time_domain_features(values)
                freq_features = _frequency_domain_features(values, nominal_freq, n_harmonics, max_freq)

                for name, value in {**time_features, **freq_features}.items():
                    record[f"{channel_prefix}__{name}"] = value

            records.append(record)

    features_df = pd.DataFrame.from_records(records)
    features_df["class_name"] = pd.Categorical(features_df["class_name"], categories=CLASS_ORDER, ordered=True)
    features_df["condition"] = pd.Categorical(features_df["condition"], categories=CONDITION_ORDER, ordered=True)

    return features_df


def get_feature_columns(features_df):
    return [column for column in features_df.columns if column not in METADATA_COLUMNS]


# Partição dos dados (nested K-Fold, sem vazamento de dados entre janelas da mesma gravação)

def _assert_no_group_leakage(groups, idx_a, idx_b):
    overlap = set(groups[idx_a]) & set(groups[idx_b])
    if overlap:
        raise AssertionError(f"Vazamento de dados detectado: grupos presentes nas duas partições: {overlap}")


def nested_group_kfold_splits(features_df, group_column="filename", target_column="class_name", outer_k=4, inner_k=3, random_state=42):
    """
    Gera uma partição nested K-Fold (StratifiedGroupKFold), garantindo que todas as
    janelas de uma mesma gravação (mesmo `group_column`, por padrão "filename")
    permaneçam sempre do mesmo lado da partição, nunca divididas entre
    treino/validação/teste.

    Com o dataset atual (4 arquivos por classe), outer_k=4 e inner_k=3 fazem cada
    fold externo e interno reservar exatamente 1 arquivo por classe para
    teste/validação, o que equivale a um leave-one-condition-out aninhado.

    Retorna uma lista (uma entrada por fold externo) de dicts:
        {
            "outer_train_idx": array de posições (iloc) em features_df,
            "outer_test_idx": array de posições (iloc) em features_df,
            "inner_splits": [(inner_train_idx, inner_val_idx), ...]  # também posições (iloc)
        }
    """
    y = features_df[target_column].astype(str).to_numpy()
    groups = features_df[group_column].to_numpy()
    row_idx = np.arange(len(features_df))

    outer_cv = StratifiedGroupKFold(n_splits=outer_k, shuffle=True, random_state=random_state)

    nested_splits = []
    for outer_train_idx, outer_test_idx in outer_cv.split(row_idx, y, groups):
        _assert_no_group_leakage(groups, outer_train_idx, outer_test_idx)

        inner_cv = StratifiedGroupKFold(n_splits=inner_k, shuffle=True, random_state=random_state)
        inner_y = y[outer_train_idx]
        inner_groups = groups[outer_train_idx]

        inner_splits = []
        for inner_train_pos, inner_val_pos in inner_cv.split(outer_train_idx, inner_y, inner_groups):
            inner_train_idx = outer_train_idx[inner_train_pos]
            inner_val_idx = outer_train_idx[inner_val_pos]
            _assert_no_group_leakage(groups, inner_train_idx, inner_val_idx)
            inner_splits.append((inner_train_idx, inner_val_idx))

        nested_splits.append({
            "outer_train_idx": outer_train_idx,
            "outer_test_idx": outer_test_idx,
            "inner_splits": inner_splits
        })

    return nested_splits


# Avaliação da capacidade discriminativa e da redundância entre features

def evaluate_feature_discriminability(features_df, feature_columns=None, target_column="class_name", random_state=42):
    """
    Avalia a capacidade de cada feature separar as quatro condições do motor,
    usando ANOVA F-value (relação linear) e informação mútua (relação geral,
    inclusive não-linear). Quanto maiores os valores, mais discriminativa a feature.
    """
    if feature_columns is None:
        feature_columns = get_feature_columns(features_df)

    X = features_df[feature_columns].to_numpy()
    y = features_df[target_column].astype(str).to_numpy()

    f_values, p_values = f_classif(X, y)
    mi_values = mutual_info_classif(X, y, random_state=random_state)

    return pd.DataFrame({
        "feature": feature_columns,
        "f_value": f_values,
        "p_value": p_values,
        "mutual_info": mi_values
    }).sort_values("mutual_info", ascending=False).reset_index(drop=True)


def plot_feature_discriminability(discriminability_df, top_n=20, metric="mutual_info"):
    """
    Gráfico de barras das features mais discriminativas em relação à condição do
    motor, segundo `metric` ("mutual_info" ou "f_value").
    """
    top = discriminability_df.sort_values(metric, ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 0.35 * len(top) + 1))
    ax.barh(top["feature"].to_numpy()[::-1], top[metric].to_numpy()[::-1], color="tab:blue")
    ax.set_xlabel(metric)
    ax.set_title(f"Top {len(top)} features mais discriminativas ({metric})")
    plt.tight_layout()
    plt.show()


def plot_feature_correlation(features_df, feature_columns=None, method="spearman"):
    """
    Matriz de correlação entre as features extraídas, para identificar redundância
    (ex.: o mesmo descritor estatístico calculado em canais diferentes tende a ser
    altamente correlacionado).
    """
    if feature_columns is None:
        feature_columns = get_feature_columns(features_df)

    corr = features_df[feature_columns].corr(method=method)

    fig, ax = plt.subplots(figsize=(0.35 * len(feature_columns) + 2, 0.35 * len(feature_columns) + 2))
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(np.arange(len(feature_columns)))
    ax.set_xticklabels(feature_columns, rotation=90, fontsize=6)
    ax.set_yticks(np.arange(len(feature_columns)))
    ax.set_yticklabels(feature_columns, fontsize=6)
    fig.colorbar(im, ax=ax, label=f"Correlação ({method})", shrink=0.8)
    ax.set_title("Matriz de correlação entre features")
    plt.tight_layout()
    plt.show()


def find_redundant_feature_pairs(features_df, feature_columns=None, method="spearman", threshold=0.9):
    """
    Lista pares de features com correlação (em módulo) acima de `threshold`,
    candidatas a redundantes, úteis para reduzir dimensionalidade antes da
    modelagem.
    """
    if feature_columns is None:
        feature_columns = get_feature_columns(features_df)

    corr = features_df[feature_columns].corr(method=method).to_numpy()
    n = len(feature_columns)

    pairs = [
        {"feature_a": feature_columns[i], "feature_b": feature_columns[j], "correlation": corr[i, j]}
        for i in range(n)
        for j in range(i + 1, n)
        if abs(corr[i, j]) >= threshold
    ]

    return pd.DataFrame(pairs).sort_values("correlation", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def select_uncorrelated_features(features_df, feature_columns=None, discriminability_df=None, target_column="class_name", method="spearman", threshold=0.9, random_state=42):
    """
    Seleção gulosa (greedy) de features não-redundantes: percorre as features em
    ordem decrescente de informação mútua com a classe e mantém cada uma apenas se
    não estiver fortemente correlacionada (|corr| >= threshold) com alguma feature
    já selecionada.

    Nota metodológica: o ranking por informação mútua usa o dataset inteiro (não
    um fold específico), é uma decisão de engenharia de features feita uma única
    vez, não uma seleção refeita dentro de cada fold de treino/teste. Deve ser
    documentada como tal (limitação metodológica), já que usa a coluna de classe
    de todas as janelas, inclusive as que depois caem no conjunto de teste de
    algum fold.

    Retorna a lista de features mantidas, em ordem de seleção (mais discriminativa
    primeiro).
    """
    if feature_columns is None:
        feature_columns = get_feature_columns(features_df)

    if discriminability_df is None:
        discriminability_df = evaluate_feature_discriminability(features_df, feature_columns=feature_columns, target_column=target_column, random_state=random_state)

    ordered_features = discriminability_df.sort_values("mutual_info", ascending=False)["feature"].tolist()
    corr = features_df[feature_columns].corr(method=method).abs()

    selected = []
    for feature in ordered_features:
        if all(corr.loc[feature, kept] < threshold for kept in selected):
            selected.append(feature)

    return selected
