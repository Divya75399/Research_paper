import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_pie_chart(df):
    if df.empty or "Prediction Result" not in df.columns:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.text(0.5, 0.5, "No saved predictions", ha="center", va="center", color="#64748b")
        ax.set_axis_off()
        fig.patch.set_facecolor('none')
        fig.tight_layout()
        return fig

    counts = df["Prediction Result"].value_counts()
    fig, ax = plt.subplots(figsize=(4, 4))
    colors_list = ["#EF5350", "#0F52BA"] if "Leukemia" in counts.index else ["#0F52BA", "#EF5350"]

    wedges, texts, autotexts = ax.pie(
        counts,
        labels=counts.index,
        autopct='%1.1f%%',
        colors=colors_list[:len(counts)],
        startangle=140,
        textprops=dict(color="#0f172a", weight="bold"),
        wedgeprops=dict(width=0.4, edgecolor='w')
    )

    plt.setp(autotexts, size=9, weight="bold", color="white")
    ax.set_title("Distribution of Cases", fontsize=11, weight="bold", color="#0F52BA")
    fig.patch.set_facecolor('none')
    fig.tight_layout()
    return fig


def make_monthly_trend_chart(df):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    if df.empty or "Date of Diagnosis" not in df.columns:
        ax.text(0.5, 0.5, "No monthly data available", ha="center", va="center", color="#64748b")
        ax.set_axis_off()
        fig.patch.set_facecolor('none')
        fig.tight_layout()
        return fig

    df_sorted = df.copy()
    dates = pd.to_datetime(df_sorted["Date of Diagnosis"], errors="coerce")
    df_sorted = df_sorted.loc[dates.notna()].copy()
    df_sorted["Month"] = dates.dt.strftime('%b %Y')
    monthly_counts = df_sorted.groupby("Month").agg(
        leukemia=("Prediction Result", lambda s: int((s == "Leukemia").sum())),
        normal=("Prediction Result", lambda s: int((s == "Normal").sum()))
    ).reset_index()

    if monthly_counts.empty:
        ax.text(0.5, 0.5, "No monthly data available", ha="center", va="center", color="#64748b")
        ax.set_axis_off()
        fig.patch.set_facecolor('none')
        fig.tight_layout()
        return fig

    months = monthly_counts["Month"]
    leuk_cases = monthly_counts["leukemia"]
    norm_cases = monthly_counts["normal"]

    ax.plot(months, leuk_cases, color="#EF5350", marker="o", linewidth=2.5, label="Leukemia Positive")
    ax.plot(months, norm_cases, color="#0F52BA", marker="s", linewidth=2.5, label="Normal Screened")

    ax.set_title("Monthly Screening Volumetrics", fontsize=11, weight="bold", color="#0F52BA")
    ax.set_xlabel("Timeline", fontsize=8.5, color="#64748b")
    ax.set_ylabel("Case Count", fontsize=8.5, color="#64748b")
    ax.grid(color='#e2e8f0', linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cbd5e1')
    ax.spines['bottom'].set_color('#cbd5e1')
    ax.tick_params(colors='#64748b', labelsize=8)
    ax.legend(frameon=False, fontsize=8)

    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    fig.tight_layout()
    return fig


def make_confusion_matrix_figure(cm, class_names):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix", fontweight="bold", fontsize=11, color="#0F52BA")
    ax.set_xlabel("Predicted Diagnosis", fontsize=9, color="#64748b")
    ax.set_ylabel("True Diagnosis", fontsize=9, color="#64748b")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=8.5, color="#0f172a")
    ax.set_yticklabels(class_names, fontsize=8.5, color="#0f172a")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > np.max(cm)/2 else "#0f172a",
                fontweight="bold", fontsize=11
            )
    fig.tight_layout()
    return fig


def make_roc_figure(fpr, tpr, roc_auc):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, color="#EF5350", linewidth=2.5, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#64748b", linewidth=1.2)
    ax.set_xlabel("False Positive Rate", fontsize=9, color="#64748b")
    ax.set_ylabel("True Positive Rate", fontsize=9, color="#64748b")
    ax.set_title("ROC Performance Curve", fontweight="bold", fontsize=11, color="#0F52BA")
    ax.grid(color='#e2e8f0', linestyle='--', linewidth=0.5)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    ax.set_facecolor('none')
    fig.patch.set_facecolor('none')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cbd5e1')
    ax.spines['bottom'].set_color('#cbd5e1')
    fig.tight_layout()
    return fig
