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

    print(df_long)

    sns.set_theme(style="whitegrid")
    g = sns.relplot(
        data=df_long,
        y="accuracy",
        x="state_size",
        hue="metric",
        style="logger.group",
        kind="line",
        markers=True,
        dashes=False,
        height=5,
        aspect=1,
    )
    g.set(xscale="log", ylabel="Accuracy", xlabel="State Size")
    for ax in g.axes.flat:
        ax.set_xscale("log", base=2)


def plot_fine_acc2clusters(
    df: pd.DataFrame,
    metric: str="valid/accuracy",
    ood_test: bool=False,
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

    # Keep only the biggest state size for each model.name, except for gistsa
    # gla_max = df_long[df_long["model.name"] == "gla"]["state_size"].max()
    # gsa_max = df_long[df_long["model.name"] == "gsa"]["state_size"].max()
    attn1_sizes = df_long[df_long["logger.group"] == "attn2-inf-pairs"]["state_size"].unique()
    attn1_size_subsets = attn1_sizes.tolist()[1:-1:2]
    attn2_sizes = df_long[df_long["logger.group"] == "attn-inf-pairs-withgist"]["state_size"].unique()
    attn2_size_subsets = attn2_sizes.tolist()[1:-1:2]
    df_long = df_long[
        (df_long["model.name"] == "gla") | 
        # ((df_long["state_size"] == gsa_max) & (df_long["model.name"] == "gsa")) | 
        ((df_long["model.name"] == "gistsa")) | 
        ((np.isin(df_long["state_size"], attn1_size_subsets)) & (df_long["logger.group"] == "attn2-inf-pairs")) |
        ((np.isin(df_long["state_size"], attn2_size_subsets)) & (df_long["logger.group"] == "attn-inf-pairs-withgist"))
    ]


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
        # Dynamic state size (for attention and gistsa)
        # gistsa: state_size = state_size * n_clusters
        # attention: state_size = state_size * seq_len (seq_len is the last number in the metric column)
        gistsa_mask = df_long["model.name"] == "gistsa"
        df_long.loc[gistsa_mask, "state_size"] = df_long.loc[gistsa_mask, "state_size"] * df_long.loc[gistsa_mask, "num_kv_pairs"]
        attn1_mask = df_long["logger.group"] == "attn2-inf-pairs"
        df_long.loc[attn1_mask, "state_size"] = df_long.loc[attn1_mask, "state_size"] / 5120 * df_long.loc[attn1_mask, "num_kv_pairs"] * 4  # 5120 is the max seq len in the inf-pair test set
        attn2_mask = df_long["logger.group"] == "attn-inf-pairs-withgist"
        df_long.loc[attn2_mask, "state_size"] = df_long.loc[attn2_mask, "state_size"] / 5120 * df_long.loc[attn2_mask, "num_kv_pairs"] * 5  # 5120 is the max seq len in the inf-pair test set
        # df_long.loc[gistattn_mask, "model.name"] = "gistAttn"

    # Log scale state size
    df_long["state_size (2^x)"] = np.log2(df_long["state_size"])#.astype(int)

    print(df_long)

    from matplotlib.colors import Normalize
    size_norm = Normalize(vmin=df_long["state_size"].min(), vmax=df_long["state_size"].max())
    
    sns.set_theme(style="whitegrid")
    g = sns.relplot(
        data=df_long,
        y="accuracy",
        x="n_shots",
        hue="logger.group",
        size="state_size",
        size_norm=size_norm,
        sizes=(20, 300),
        kind="scatter",
        markers=True,
        dashes=False,
        height=5,
        aspect=1,
        alpha=0.75
    )
    g.set(xscale="log", ylabel="Accuracy", xlabel="#KV Pairs")
    for ax in g.axes.flat:
        ax.set_xscale("log", base=2)


def plot_fine_acc2shots(
    df: pd.DataFrame,
    metric: str="valid/accuracy",
    n_clusters: int=4
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

    # Filter incomplete runs (e.g. attn d_model = 8)
    # attn_sizes = df_long[df_long["model.name"] == "attention"]["state_size"].unique()
    # df_long = df_long[(df_long["model.name"] != "attention") | (df_long["state_size"] != attn_sizes.min())]
    # gistsa_sizes = df_long[df_long["model.name"] == "gistsa"]["state_size"].unique()
    # df_long = df_long[(df_long["model.name"] != "gistsa") | (df_long["state_size"] != gistsa_sizes.min())]

    # Extract n_clusters, n_dims, and num_kv_pairs from the metric column
    df_long["n_clusters"] = df_long["metric"].str.extract(r'(\d+)\)$').astype(int)
    df_long["n_dims"] = df_long["metric"].str.extract(r',\s+(\d+),').astype(int)
    df_long["num_kv_pairs"] = df_long["metric"].str.extract(r'\((\d+)').astype(int)
    df_long["n_shots"] = df_long["num_kv_pairs"] / df_long["n_clusters"]
    # Dynamic state size (for attention and gistsa)
    # gistsa: state_size = state_size * n_clusters
    # attention: state_size = state_size * seq_len (seq_len is the last number in the metric column)
    gistsa_mask = df_long["model.name"] == "gistsa"
    df_long.loc[gistsa_mask, "state_size"] = df_long.loc[gistsa_mask, "state_size"] * df_long.loc[gistsa_mask, "n_clusters"]
    gistattn_mask = df_long["model.name"] == "attention"
    df_long.loc[gistattn_mask, "state_size"] = df_long.loc[gistattn_mask, "state_size"] / 12288 * df_long.loc[gistattn_mask, "num_kv_pairs"] * 3 * 2

    # Log scale state size
    df_long["state_size (2^x)"] = np.log2(df_long["state_size"])#.astype(int)

    # Filter n_clusters
    df_long = df_long[df_long["n_clusters"] == n_clusters]


    print(df_long)

    from matplotlib.colors import Normalize
    size_norm = Normalize(vmin=df_long["state_size"].min(), vmax=df_long["state_size"].max())
    
    sns.set_theme(style="whitegrid")
    g = sns.relplot(
        data=df_long,
        y="accuracy",
        x="n_shots",
        hue="model.name",
        # hue="logger.group",
        style="logger.group",
        size="state_size",
        size_norm=size_norm,
        sizes=(20, 300),
        kind="scatter",
        markers=True,
        dashes=False,
        height=5,
        aspect=1,
        alpha=0.75
    )
    g.set(xscale="log", ylabel="Accuracy", xlabel="#Shots")
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
        # sweep_id=[
            # gistsa
            # "mvqar76fbae",
            # gla
            # "mvqardcaefe",
            # "mvqardd4c5b",
            # gsa
            # "mvqard460ff",
            # "mvqarf58603",
            # gistattn
            # "mvqarc29460",
        # ],
        # inf-pairs
        # sweep_id=[
            # gla
            # "inf49e4c2",
            # gla with gist,
            # "infb3b888",
            # attn
            # "inf365da5",
            # "inf99e8dd",
            # attn with gist,
            # "inf8829ac",
            # "infbf94f2",
            # gistsa
            # "inf4a8212",
        # ],
        # inf-shots
        # sweep_id=[
        #     # gla
        #     "infshots307426",
        #     "infshots3c677f",
        #     "infshots508fd0",
        #     "infshots72b6ad",
        #     "infshotsb9f6e4",
        #     "infshotsf29885",
        #     # gistsa
        #     "infshots051d4c",
        #     "infshots2b2945",
        #     "infshots5eea7b",
        #     "infshotsea888e",
        #     "infshotsf16cf2",
        #     # attn
        #     "infshots2f2acd",
        #     "infshotse0dc80",
        #     "infshotse7cab2",
        #     # gdn
        #     "infshots599f5a",
        #     "infshotsb21462",
        #     "infshotsb59585",
        # ],
        # inf-shots-random_gists
        # sweep_id=[
        #     "infshotsf06b3b",  # random
        #     "infshots52f5a1"  # label
        # ],
        # gist vs gistsa ^ labelGists vs extraGists
        sweep_id=[
            "infshots10d597",  # extraGists
            "infshots66629b",  # labelGists
            "infshotse66661",
            "infshotsc49a8e",  # labelGists (first occurence)
            "infshotsdb7b79",
            "selfproto2a07c4",  # selfProto-lite
        ],
        project_name="zoology"
    )

    print(df)

    # Replace logger.group == "bijective" with "deltaNet-mqar-bijective"
    # df["logger.group"] = df["logger.group"].replace("bijective", "deltaNet-mqar-bijective")

    # df = df[df["state_size"] > 2 ** 12]

    # plot(df=df)
    # 2d2c
    # plt.savefig("results/inf-shots.pdf")
    # State-size sweep for gla and gsa
    # plt.savefig("results/mqar-gla-sweep.pdf")
    # Attention sweep
    # plt.savefig("results/attn.pdf")

    # plot_fine(df=df, ood_test=False)
    # plot_fine_acc2clusters(df=df, ood_test=False)
    for n_clusters in [2, 4]:
        plot_fine_acc2shots(df=df, n_clusters=n_clusters)
        plt.savefig(f"results/gistExp-nclusters{n_clusters}-fine.pdf")
    # State-size sweep for gla and gsa
    # plt.savefig("results/mqar-gla-sweep_fine.pdf")
    # Attention sweep
    # plt.savefig("results/attn_fine.pdf")

    print("Done")
