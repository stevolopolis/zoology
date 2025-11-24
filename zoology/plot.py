import os

import pandas as pd
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from zoology.analysis.utils import fetch_wandb_runs


def plot(
    df: pd.DataFrame,
    metric: str="valid/accuracy",
):
    idx = df.groupby(
        ["state_size", "model.name", "logger.group"]
    )[metric].idxmax(skipna=True).dropna()
    df = df.loc[idx]
    plot_df = df

    sns.set_theme(style="whitegrid")
    g = sns.relplot(
        data=plot_df,
        y=metric,
        x="state_size",
        # hue="model.name",
        hue="logger.group",
        kind="line",
        markers=True,
        height=5,
        aspect=1,
    )
    g.set(xscale="log", ylabel="Accuracy", xlabel="State Size")
    for ax in g.axes.flat:
        ax.set_xscale("log", base=2)


def plot_fine(
    df: pd.DataFrame,
    metric: str="valid/accuracy",
    ood_test: bool=False
):
    model_columns = [c for c in df.columns if "accuracy-" in c]
    idx = df.groupby(
        ["state_size", "model.name", "logger.group"]
    )[metric].idxmax(skipna=True).dropna()
    df = df.loc[idx]
    # plot_df = df

    df_long = df.melt(
        id_vars=["state_size", "model.name", "logger.group"],
        value_vars=model_columns,
        var_name="metric",
        value_name="accuracy",
    )
    # Extract n_clusters, n_dims, and num_kv_pairs from the metric column
    if ood_test:
        df_long["n_clusters"] = df_long["metric"].str.extract(r'(\d+)\)$').astype(int)
        df_long["n_dims"] = df_long["metric"].str.extract(r',\s+(\d+),').astype(int)
        df_long["num_kv_pairs"] = df_long["metric"].str.extract(r'\((\d+)').astype(int)
        # Dynamic state size (for attention and gistsa)
        # gistsa: state_size = state_size * n_clusters
        # attention: state_size = state_size * seq_len (seq_len is the last number in the metric column)
        gistsa_mask = df_long["model.name"] == "gistsa"
        df_long.loc[gistsa_mask, "state_size"] = df_long.loc[gistsa_mask, "state_size"] * df_long.loc[gistsa_mask, "n_clusters"]
        gistattn_mask = df_long["model.name"] == "attention"
        df_long.loc[gistattn_mask, "state_size"] = df_long.loc[gistattn_mask, "state_size"] * df_long.loc[gistattn_mask, "num_kv_pairs"] * (df_long.loc[gistattn_mask, "n_dims"] + 1) * 2
        df_long.loc[gistattn_mask, "model.name"] = "gistAttn"
    else:
        df_long["num_kv_pairs"] = df_long["metric"].str.extract(r'(\d+)$').astype(int)

    # Log scale state size
    # df_long["state_size"] = np.log2(df_long["state_size"])

    # Keep only the biggest state size for each model.name, except for gistsa
    gla_max = df_long[df_long["model.name"] == "gla"]["state_size"].max()
    gsa_max = df_long[df_long["model.name"] == "gsa"]["state_size"].max()
    gistAttn_max = df_long[df_long["model.name"] == "gistAttn"]["state_size"].max()
    df_long = df_long[
        ((df_long["state_size"] == gla_max) & (df_long["model.name"] == "gla")) | ((df_long["state_size"] == gsa_max) & (df_long["model.name"] == "gsa")) | ((df_long["model.name"] == "gistsa")) | ((df_long["state_size"] == gistAttn_max) & (df_long["model.name"] == "gistAttn"))
    ]

    print(df_long)

    sns.set_theme(style="whitegrid")
    g = sns.relplot(
        data=df_long,
        y="accuracy",
        x="n_clusters",
        # compare across metrics
        hue="model.name",
        style="n_dims",
        size="state_size",
        # compare across models
        # hue="logger.group",
        # hue_order=sorted(df_long["logger.group"].unique()),
        # style="metric",
        kind="scatter",
        markers=True,
        dashes=False,
        height=5,
        aspect=1,
    )
    g.set(xscale="log", ylabel="Accuracy", xlabel="#Clusters")
    for ax in g.axes.flat:
        ax.set_xscale("log", base=2)


if __name__ == "__main__" :
    df = fetch_wandb_runs(
        # GSA VS GLA
        # sweep_id=[
        #     "surjectivea670d1",
        #     "surjectived1b063",
        #     "surjective9c42cf",
        #     "surjective5e255f"
        # ],
        # MQGAR surjective vs bijective
        # sweep_id=[
        #     "mqgardea847",
        #     "mqgarf36fd3",
        #     "mqgar854857"
        # ],
        # MQGAR 4, 8, 12 tokens + MQAR
        # sweep_id=[
        #     "mqgardea847",
        #     "mqgar25c7e0",
        #     "mqgaraddf33",
        #     "mqgar344dbd",
        #     "bijective4f174a",
        #     "mqgarc29f22"
        # ],
        # State-size sweep for gla and gsa
        # sweep_id=[
            # gla
            # "surjective8e6eb8",
            # "surjectivee8e43b",
            # "surjective86c4e9",
            # "surjective7a6d77",
            # "surjectivee3606f",
            # gsa
            # "surjective609a18",
            # "surjectivece73cd",
            # "surjectiveb1f919",
        # ],
        # Sharing n last key tokens
        # sweep_id=[
        #     "mqgarc29f22",
        #     "mqgar13942b",
        #     "mqgareeeba5",
        #     "mqgar9b7093",
        #     "mqgardea847"
        # ],
        # Multi-token tracking
        # sweep_id=[
        #     "mqgar0ffc8d",
        #     "mqgardea847",
        #     "mqgar6fff2a"
        # ],
        # Attention sweep
        # sweep_id=[
        #     "surjective3af9b0"
        # ],
        # gla multi layer
        # sweep_id=[
        #     "surjective8053c8"
        # ],
        # gla 2d2c 
        # sweep_id=[
            # 2d2c
            # "mvqar68f2e8",
            # 2d2c-test-mqar
            # "mvqarbe2a3f",
            # 2d2c-test-mqgar
            # "mvqar4e13a2",
            # mqar
        #     "surjective7a6d77",
        #     "surjective86c4e9",
        #     "surjective8e6eb8",
        #     "surjectivee3606f"
        # ],
        # gistsa ood
        sweep_id=[
            # gistsa
            "mvqar76fbae",
            # gla
            "mvqardcaefe",
            "mvqardd4c5b",
            # gsa
            "mvqard460ff",
            "mvqarf58603",
            # gistattn
            "mvqarc29460",
        ],
        project_name="zoology"
    )

    print(df)

    # Replace logger.group == "bijective" with "deltaNet-mqar-bijective"
    # df["logger.group"] = df["logger.group"].replace("bijective", "deltaNet-mqar-bijective")

    # df = df[df["state_size"] > 2 ** 12]

    plot(df=df)
    # 2d2c
    plt.savefig("results/gistsa2-2d2c-ood.pdf")
    # State-size sweep for gla and gsa
    # plt.savefig("results/mqar-gla-sweep.pdf")
    # Attention sweep
    # plt.savefig("results/attn.pdf")

    plot_fine(df=df, ood_test=True)
    # 2d2c
    plt.savefig("results/gistsa2-2d2c-ood_fine.pdf")
    # State-size sweep for gla and gsa
    # plt.savefig("results/mqar-gla-sweep_fine.pdf")
    # Attention sweep
    # plt.savefig("results/attn_fine.pdf")

    print("Done")
