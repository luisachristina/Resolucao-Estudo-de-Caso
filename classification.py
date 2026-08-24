"""
Neste arquivo serão armazenadas as configurações e funções auxilirares para a classificações de condições do motor.

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score, average_precision_score
)

from eda_functionc import CLASS_ORDER

RANDOM_STATE = 42


# Configuração dos dois modelos e das grades de hiperparâmetros a testar.
# needs_scaling indica se o modelo precisa de features padronizadas (SVM, por
# ser baseado em distância) ou não (Random Forest, invariante a escala).

MODEL_REGISTRY = {
    "random_forest": {
        "label": "Random Forest",
        "build": lambda params: RandomForestClassifier(
            random_state=RANDOM_STATE, class_weight="balanced", **params
        ),
        "param_grid": [
            {"n_estimators": n_estimators, "max_depth": max_depth, "min_samples_leaf": min_samples_leaf}
            for n_estimators in [100, 300]
            for max_depth in [None, 8, 16]
            for min_samples_leaf in [1, 2, 4]
        ],
        "needs_scaling": False
    },
    "svm_rbf": {
        "label": "SVM (kernel RBF)",
        # CalibratedClassifierCV envolve o SVC para fornecer predict_proba (o
        # parâmetro probability=True do SVC está obsoleto a partir do scikit-learn
        # 1.9). cv=3 e ensemble=False mantêm o custo computacional baixo.
        "build": lambda params: CalibratedClassifierCV(
            estimator=SVC(kernel="rbf", random_state=RANDOM_STATE, class_weight="balanced", **params),
            method="sigmoid", cv=3, ensemble=False
        ),
        "param_grid": [
            {"C": c, "gamma": gamma}
            for c in [0.1, 1, 10, 100]
            for gamma in ["scale", 0.001, 0.01, 0.1]
        ],
        "needs_scaling": True
    }
}


# Treino e predição de uma combinação de hiperparâmetros

def _fit_predict(model_key, params, X_train, y_train, X_eval):
    """
    Treina o modelo `model_key` com os hiperparâmetros `params` em (X_train,
    y_train) e retorna o modelo treinado, as predições de classe e as
    probabilidades por classe para X_eval. Quando o modelo exige escala (SVM), o
    StandardScaler é ajustado apenas em X_train, nunca em X_eval, para não vazar
    estatísticas do conjunto de avaliação para o treino.
    """
    spec = MODEL_REGISTRY[model_key]
    model = spec["build"](params)

    if spec["needs_scaling"]:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_eval = scaler.transform(X_eval)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)

    return model, y_pred, y_proba


# Busca de hiperparâmetros usando os folds internos (nunca o teste externo)

def tune_hyperparameters(features_df, feature_columns, fold, model_key, target_column="class_name"):
    """
    Faz busca em grade nos hiperparâmetros de `model_key`, usando os folds internos
    de `fold["inner_splits"]` (índices posicionais em `features_df`, gerados por
    `nested_group_kfold_splits`). Para cada combinação, calcula o F1-macro médio
    entre os folds internos e retorna a combinação com maior média.

    O F1-macro foi escolhido como critério porque, com classes balanceadas mas em
    um problema de diagnóstico de falhas, ele penaliza igualmente o desempenho
    ruim em qualquer uma das quatro classes, diferente da acurácia simples, que
    pode mascarar uma classe minoritária mal classificada.
    """
    spec = MODEL_REGISTRY[model_key]
    best_params, best_score = None, -np.inf

    for params in spec["param_grid"]:
        inner_scores = []

        for inner_train_idx, inner_val_idx in fold["inner_splits"]:
            X_train = features_df.iloc[inner_train_idx][feature_columns].to_numpy()
            y_train = features_df.iloc[inner_train_idx][target_column].astype(str).to_numpy()
            X_val = features_df.iloc[inner_val_idx][feature_columns].to_numpy()
            y_val = features_df.iloc[inner_val_idx][target_column].astype(str).to_numpy()

            _, y_pred, _ = _fit_predict(model_key, params, X_train, y_train, X_val)
            inner_scores.append(f1_score(y_val, y_pred, average="macro"))

        mean_score = float(np.mean(inner_scores))
        if mean_score > best_score:
            best_score, best_params = mean_score, params

    return best_params, best_score


# Métricas de área sob a curva (ROC-AUC e PR-AUC), one-vs-rest

def _auc_metrics(y_true, y_proba, classes):
    """
    Calcula ROC-AUC macro e PR-AUC (average precision) macro, one-vs-rest, para um
    problema multiclasse. PR-AUC costuma ser mais informativa que ROC-AUC quando
    há interesse em identificar corretamente a classe positiva (a falha) sem se
    deixar enganar por um grande número de verdadeiros negativos.
    """
    y_true_bin = label_binarize(y_true, classes=classes)

    pr_auc_per_class = {
        cls: average_precision_score(y_true_bin[:, i], y_proba[:, i])
        for i, cls in enumerate(classes)
    }
    roc_auc_macro = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=classes)
    pr_auc_macro = float(np.mean(list(pr_auc_per_class.values())))

    return {
        "roc_auc_macro": roc_auc_macro,
        "pr_auc_macro": pr_auc_macro,
        "pr_auc_per_class": pr_auc_per_class
    }


# Validação nested completa: busca de hiperparâmetros + avaliação no teste externo

def run_nested_cross_validation(features_df, feature_columns, nested_splits, model_key, target_column="class_name"):
    """
    Roda a validação nested completa para `model_key`: em cada fold externo,
    escolhe hiperparâmetros via busca em grade nos folds internos, treina de novo
    no conjunto de treino externo completo com os melhores hiperparâmetros, e
    avalia no conjunto de teste externo (arquivos nunca vistos, nem no treino, nem
    na escolha de hiperparâmetros).

    Retorna uma lista com um dict de resultados por fold externo.
    """
    fold_results = []

    for fold_id, fold in enumerate(nested_splits):
        best_params, inner_f1_macro = tune_hyperparameters(features_df, feature_columns, fold, model_key, target_column)

        X_train = features_df.iloc[fold["outer_train_idx"]][feature_columns].to_numpy()
        y_train = features_df.iloc[fold["outer_train_idx"]][target_column].astype(str).to_numpy()
        X_test = features_df.iloc[fold["outer_test_idx"]][feature_columns].to_numpy()
        y_test = features_df.iloc[fold["outer_test_idx"]][target_column].astype(str).to_numpy()

        model, y_pred, y_proba = _fit_predict(model_key, best_params, X_train, y_train, X_test)
        auc_metrics = _auc_metrics(y_test, y_proba, model.classes_)

        fold_results.append({
            "model_key": model_key,
            "fold": fold_id,
            "best_params": best_params,
            "inner_f1_macro": inner_f1_macro,
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "roc_auc_macro": auc_metrics["roc_auc_macro"],
            "pr_auc_macro": auc_metrics["pr_auc_macro"],
            "pr_auc_per_class": auc_metrics["pr_auc_per_class"],
            "y_true": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "classes": model.classes_
        })

    return fold_results


def train_and_compare_models(features_df, feature_columns, nested_splits, model_keys=("random_forest", "svm_rbf"), target_column="class_name"):
    """
    Roda `run_nested_cross_validation` para cada modelo em `model_keys` e retorna
    um dict {model_key: fold_results}.
    """
    return {
        model_key: run_nested_cross_validation(features_df, feature_columns, nested_splits, model_key, target_column)
        for model_key in model_keys
    }


# Agregação e apresentação dos resultados

def summarize_nested_cv_results(all_results):
    """
    Recebe o dict retornado por `train_and_compare_models` e monta uma tabela
    (uma linha por modelo) com média e desvio padrão, entre os folds externos, de
    cada métrica.
    """
    metrics = ["accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc_macro", "pr_auc_macro"]
    rows = []

    for model_key, fold_results in all_results.items():
        row = {"model": MODEL_REGISTRY[model_key]["label"]}
        for metric in metrics:
            values = [r[metric] for r in fold_results]
            row[f"{metric}_mean"] = np.mean(values)
            row[f"{metric}_std"] = np.std(values)
        rows.append(row)

    return pd.DataFrame(rows)


def concatenate_fold_predictions(fold_results):
    """
    Concatena as predições de todos os folds externos (conjuntos de teste
    disjuntos), permitindo um relatório único de classificação sobre o dataset
    inteiro obtido via validação cruzada.
    """
    y_true = np.concatenate([r["y_true"] for r in fold_results])
    y_pred = np.concatenate([r["y_pred"] for r in fold_results])
    return y_true, y_pred


def per_class_report(fold_results, labels=None):
    """
    Relatório de precision/recall/f1 por classe, calculado sobre as predições
    concatenadas de todos os folds externos.
    """
    if labels is None:
        labels = CLASS_ORDER

    y_true, y_pred = concatenate_fold_predictions(fold_results)
    report = classification_report(y_true, y_pred, labels=labels, target_names=labels, output_dict=True, zero_division=0)

    return pd.DataFrame(report).T


def aggregate_confusion_matrix(fold_results, labels=None):
    """
    Soma as matrizes de confusão de todos os folds externos, já que os conjuntos
    de teste de cada fold são disjuntos (arquivos diferentes).
    """
    if labels is None:
        labels = CLASS_ORDER

    total_cm = np.zeros((len(labels), len(labels)), dtype=int)
    for fold_result in fold_results:
        total_cm += confusion_matrix(fold_result["y_true"], fold_result["y_pred"], labels=labels)

    return pd.DataFrame(total_cm, index=labels, columns=labels)


def plot_confusion_matrix(cm_df, title="Matriz de confusão agregada nos folds"):
    matrix = cm_df.to_numpy()

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(cm_df.columns)))
    ax.set_xticklabels(cm_df.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(cm_df.index)))
    ax.set_yticklabels(cm_df.index)
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    ax.set_title(title)

    threshold = matrix.max() / 2
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > threshold else "black")

    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.show()


def plot_model_comparison(summary_df, metric="f1_macro"):
    """
    Gráfico de barras comparando os modelos em `summary_df` (saída de
    `summarize_nested_cv_results`) numa métrica, com barra de erro dada pelo
    desvio padrão entre folds externos.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(
        summary_df["model"],
        summary_df[f"{metric}_mean"],
        yerr=summary_df[f"{metric}_std"],
        capsize=5,
        color=["tab:blue", "tab:orange"]
    )
    ax.set_ylabel(metric)
    ax.set_title(f"Comparação dos modelos: {metric} (media entre folds, barra = desvio padrao)")
    plt.tight_layout()
    plt.show()


def plot_metric_by_fold(all_results, metric="f1_macro"):
    """
    Mostra a evolução de uma métrica fold a fold, para os dois modelos, útil para
    verificar se algum fold (ou seja, alguma condição de operação específica)
    é sistematicamente mais difícil de classificar.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    for model_key, fold_results in all_results.items():
        values = [r[metric] for r in fold_results]
        ax.plot(range(len(values)), values, marker="o", label=MODEL_REGISTRY[model_key]["label"])

    ax.set_xlabel("Fold externo")
    ax.set_ylabel(metric)
    ax.set_xticks(range(len(next(iter(all_results.values())))))
    ax.legend()
    ax.set_title(f"{metric} por fold externo")
    plt.tight_layout()
    plt.show()
