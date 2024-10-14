import argparse
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm

# import seaborn as sns

KS = [50, 100, 200, 100000]


def execute(input, graph_folder, k):
    df = pd.read_csv(input, sep=";")
    df = df.loc[~(df == 0).all(axis=1)]
    if not os.path.exists(graph_folder):
        os.makedirs(graph_folder)
    frequency_column = "frequency"
    other_metrics = [
        metric for metric in df.columns[1:] if metric != frequency_column
    ]
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(
        df["prefix_length"][:k],
        df[frequency_column][:k],
        "g-",
        label=frequency_column,
    )
    ax1.set_xlabel("Prefix Length")
    ax1.set_ylabel(f"Frequency ({frequency_column})", color="g")
    ax1.tick_params(axis="y", labelcolor="g")
    ax2 = ax1.twinx()

    for metric in other_metrics:
        ax2.plot(df["prefix_length"][:k], df[metric][:k], label=metric)

    ax2.set_ylabel("Metrics (0-1)", color="b")
    ax2.tick_params(axis="y", labelcolor="b")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right")

    plt.title("Metrics vs Prefix Length - " + input)

    plt.savefig(
        os.path.join(graph_folder, f"all_metrics_vs_prefix_length_{k}.png")
    )
    #delete the plot
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    for k in tqdm(KS):
        execute(args.input, args.output_dir, k)
