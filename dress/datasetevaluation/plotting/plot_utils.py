from typing import Tuple, Union, List
import warnings
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib import patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.metrics import silhouette_samples, silhouette_score
import pandas as pd
from dna_features_viewer import GraphicFeature, GraphicRecord
import numpy as np
import seaborn as sns
import io


def buffered_ax(filename: str, ax: matplotlib.axes._axes.Axes) -> dict:
    """Returns a buffered ax to be later written to disk

    Args:
        filename (str): Name of the filename
        ax (matplotlib.axes._axes.Axes): Ax to be buffered

    Returns:
        dict: Output dict
    """
    buffer = io.BytesIO()
    plt.savefig(buffer, format="pdf")
    plt.close()
    buffer.seek(0)
    return {filename: buffer}


def y_density_plot(
    ax: matplotlib.axes._axes.Axes, y: np.ndarray, cluster_id: int, n_clusters: int
) -> matplotlib.axes._axes.Axes:
    """
    Generate a distribution of the labels within each cluster

    Args:
        ax (matplotlib.axes._axes.Axes): Matplotlib ax.
        y (np.ndarray): The labels of the samples.
        cluster_id (int): The ID of the cluster to plot.
        n_clusters (int): The number of formed clusters.

    Returns:
        matplotlib.axes._axes.Axes:  A Matplotlib Axes object with the density plot.
    """

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    color = cm.nipy_spectral(float(cluster_id) / n_clusters)

    sns.kdeplot(
        data=y,
        fill=True,
        color=color,
        alpha=0.7,
        edgecolor="black",
        ax=ax,
    )

    ax.set_xlim([0, 1])

    line = Line2D([], [], color=color, linewidth=1)
    ax.legend(
        handles=[line],
        labels=[f"{cluster_id} (N={len(y)})"],
        bbox_to_anchor=(1.3, 1),
        loc="upper right",
        frameon=False,
        # prop={"size": 8},
    )
    ax.set_xlabel("Model score")
    return ax


def silhouette_plot(
    ax: matplotlib.axes._axes.Axes,
    X: Union[pd.DataFrame, np.ndarray],
    estimator,
    n_clusters: int,
) -> Tuple[matplotlib.axes._axes.Axes, float]:
    """
    Generate a silhouette plot in a cluster analysis.
    Code adapted from: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

    Args:
        ax (matplotlib.axes._axes.Axes): Matplotlib ax.
        X (Union[pd.DataFrame, np.ndarray]): The input data used to perform the clustering.
        n_clusters (int): The number of clusters formed.
        estimator (Clustering, None]): The clustering estimator used.

    Returns:
        matplotlib.axes._axes.Axes:  A Matplotlib Axes object with the silhouette plot.
        float: The average silhouette score for all samples.
    """
    if estimator is None:
        return ax

    # Insert blanck space between silhouette
    ax.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    cluster_labels = estimator.labels_
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        silhouette_avg = silhouette_score(X, cluster_labels)

    # Compute the silhouette scores for each sample
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

    # Plot the silhouette per cluster
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax.set_xlabel("Silhouette coefficient values")
    ax.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    # ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    return ax, silhouette_avg


def scatterplot(
    ax: matplotlib.axes._axes.Axes,
    data: np.ndarray,
    labels: np.ndarray,
    draw_legend: bool,
) -> matplotlib.axes._axes.Axes:
    """
    Plot the datapoints in 2D space.

    Args:
        ax (matplotlib.axes._axes.Axes): Matplotlib ax.
        data (np.ndarray): The data (after performing dimensionality reduction) to plot.
        labels (np.ndarray): The labels of the data.

    Returns:
        matplotlib.axes._axes.Axes: A Matplotlib Axes object with a scatter plot with the clusters.
    """
    _colors = cm.get_cmap("coolwarm")
    norm = mcolors.Normalize(vmin=labels.min(), vmax=labels.max())
    scalar_map = cm.ScalarMappable(norm=norm, cmap=_colors)
    scalar_map.set_clim(0, 1)

    ax.scatter(
        data[:, 0],
        data[:, 1],
        marker=".",
        s=30,
        lw=0,
        alpha=0.7,
        color=scalar_map.to_rgba(labels),
        edgecolor="k",
    )

    if draw_legend:
        cbar = plt.colorbar(scalar_map, ax=ax, pad=0.1, shrink=0.5)
        cbar.ax.set_title("Model", fontsize=9)
        cbar.ax.tick_params(labelsize=9)
        cbar.ax.set_ylabel("", fontsize=0)
    return ax


def draw_gene_structure(
    ss_idx: list,
    seq_len: int,
    score: float | None = None,
    perturbations: list | None = None,
    zoom_in: bool = False,
    ax: matplotlib.axes._axes.Axes | None = None,
):
    """ """

    features = []
    upstream = ss_idx[0]
    if all(coord in ["<NA>", np.nan] for coord in upstream):
        features.append(GraphicFeature(start=0, end=0, color="#ffd700"))

    elif any(coord in ["<NA>", np.nan] for coord in upstream):
        features.append(
            GraphicFeature(start=0, end=upstream[1], color="#ffd700", open_left=True)
        )

    else:
        features.append(
            GraphicFeature(start=upstream[0], end=upstream[1], color="#ffd700")
        )

    cassette = ss_idx[1]
    features.append(GraphicFeature(start=cassette[0], end=cassette[1], color="#ffcccc"))
    if score:
        features.append(
            GraphicFeature(
                start=(cassette[1] + cassette[0]) // 2,
                end=(cassette[1] + cassette[0]) // 2,
                fontdict={"fontsize": 8},
                color="#ffcccc",
                label=f"Score: {score}",
            )
        ),
    downstream = ss_idx[2]
    if all(coord in ["<NA>", np.nan] for coord in downstream):
        features.append(GraphicFeature(start=seq_len, end=seq_len, color="#ffd700"))

    elif any(coord in ["<NA>", np.nan] for coord in downstream):
        features.append(
            GraphicFeature(
                start=downstream[0],
                end=seq_len,
                color="#ffd700",
                open_right=True,
            )
        )

    else:
        features.append(
            GraphicFeature(start=downstream[0], end=downstream[1], color="#ffd700")
        )

    if perturbations:
        for p in perturbations:
            start = p[1]
            if p[0] == "SNV":
                color = "grey"
                end = start

            elif p[0] == "RandomInsertion":
                color = "darkblue"
                end = start

            elif p[0] == "RandomDeletion":
                color = "darkred"
                end = start + len(p[2]) - 1

            label = f"{p[0]}({p[2]})"
            features.append(
                GraphicFeature(
                    start=start,
                    end=end,
                    color=color,
                    label=label,
                    fontdict={"fontsize": 8},
                )
            )

    record = GraphicRecord(
        sequence_length=seq_len,
        features=features,
    )

    if zoom_in:
        record = record.crop((cassette[0] - 100, cassette[1] + 100))

    return record.plot(ax=ax) if ax else record.plot()


def create_gridSpec(
    n_groups: int, is_different_original_seq: bool
) -> Tuple[plt.Figure, List[matplotlib.axes._axes.Axes]]:
    """
    Create a grid spec based on the number of groups
    to be displayed

    Args:
        n_groups (int): Number of groups to be displayed
        is_different_original_seq (bool): Whether the original sequence
            is different same when dataset is paired (`n_groups` == 2)

    Returns:
        Tuple[plt.Figure, List[matplotlib.axes._axes.Axes]]
    """
    axes = []
    if n_groups == 1:
        n_rows = 2
        ax_heights = [1.25, 1]
        fig = plt.figure(figsize=(8, 3.5))

    elif n_groups == 2 and is_different_original_seq:
        n_rows = 4
        ax_heights = [1.25, 1, 1.25, 1]
        fig = plt.figure(figsize=(10, 6))

    elif n_groups == 2:
        n_rows = 3
        ax_heights = [1.25, 1, 1.25]
        fig = plt.figure(figsize=(9, 5))

    else:
        raise NotImplementedError("Only 1 or 2 groups are supported")

    spec = gridspec.GridSpec(
        nrows=n_rows, ncols=1, height_ratios=ax_heights, figure=fig
    )

    ax1 = fig.add_subplot(spec[0])
    ax2 = fig.add_subplot(spec[1], sharex=ax1)
    axes.extend([ax1, ax2])
    if n_rows == 3:
        ax3 = fig.add_subplot(spec[2], sharex=ax1)
        axes.append(ax3)
        axes[1].set_position(
            [
                axes[1].get_position().x0,
                axes[1].get_position().y0 + 0.06,
                axes[1].get_position().width,
                axes[1].get_position().height,
            ]
        )
    elif n_rows == 4:
        ax3 = fig.add_subplot(spec[2])
        ax4 = fig.add_subplot(spec[3], sharex=ax3)
        axes.extend([ax3, ax4])

        axes[1].set_position(
            [
                axes[1].get_position().x0,
                axes[1].get_position().y0 + 0.05,
                axes[1].get_position().width,
                axes[1].get_position().height,
            ]
        )

        axes[2].set_position(
            [
                axes[2].get_position().x0,
                axes[2].get_position().y0 - 0.05,
                axes[2].get_position().width,
                axes[2].get_position().height,
            ]
        )
    return fig, axes


def draw_position_dist(
    ax: matplotlib.axes._axes.Axes,
    df: pd.DataFrame,
    group_name: str,
    seq_id: str,
    seq_len: int,
    ss_idx: list,
    score: float,
    split_effects: bool = True,
    split_seeds: bool = True,
    draw_exon_boundaries: bool = True,
    draw_legend: bool = True,
    zoom_in: bool = False,
):
    """ """

    def _add_text_labels(
        df: pd.DataFrame,
        ax: matplotlib.axes._axes.Axes,
        x_pos: float = 0.79,
        y_pos: float = 0.975,
        split: bool = False,
    ):
        """
        Add number of individuals and number of mutations present in the plot

        Args:
            df (pd.DataFrame): Dataframe containing the plotted data
            ax (matplotlib.axes._axes.Axes): Axes object to plot on
            x_pos (int, optional): x position of the text. Defaults to 0.85.
            y_pos (int, optional): y position of the text. Defaults to 0.975.
            split (bool, optional): Whether the plot is split by model effect. Defaults to False.
        """
        if split:
            n_perturb = df.total_n_perturb_split.iloc[0]
        else:
            n_perturb = df.total_n_perturb.iloc[0]
        ax.text(
            x_pos,
            y_pos,
            "n_ind={}\nn_mut={}".format(len(df.id.unique()), n_perturb),
            ha="left",
            va="top",
            fontsize=8,
            transform=ax.transAxes,
        )

    def _default_hist(
        ax: matplotlib.axes._axes.Axes,
        df: pd.DataFrame,
        seq_len: int,
        zoom_in: bool,
        draw_legend: bool,
        add_annotation: bool,
    ):
        """
        Draw a default hist plot without splitting by Model effect

        Args:
            df (pd.DataFrame): Dataframe containing the data to be plotted
            ax (matplotlib.axes._axes.Axes): Axes object to plot on
            seq_len (int): Length of the sequence
            zoom_in (bool, optional): Whether to zoom in on the plot.
            draw_legend (bool, optional): Whether to draw the legend.
            add_annotation (bool, optional): Whether to add text labels.
        """

        if zoom_in:
            assert "model_effect" in df.columns
            pal = sns.color_palette("deep")
            sns.histplot(
                data=df,
                x="positions",
                binwidth=1,
                element="bars",
                kde=False,
                fill=True,
                hue="model_effect",
                multiple="stack",
                ax=ax,
                palette={"Splicing Up": pal[0], "Splicing Down": pal[1]},
            )
            legend = ax.get_legend()
            legend.remove()
            if draw_legend:
                ax.legend(
                    [mpatches.Patch(color=pal[0]), mpatches.Patch(color=pal[1])],
                    ["Splicing up", "Splicing down"],
                    title="",
                    loc="center right",
                    bbox_to_anchor=(1.1, 1.1),
                    fontsize="small",
                    handleheight=0.5,
                    handlelength=0.7,
                    frameon=False,
                    borderaxespad=0,
                )

        else:
            sns.histplot(
                data=df,
                x="positions",
                binwidth=seq_len * 0.005,
                element="bars",
                kde=False,
                fill=True,
                ax=ax,
            )

        if add_annotation:
            _add_text_labels(df, ax, x_pos=0.95, y_pos=0.9)

    def _split_seeds_hist(
        ax: matplotlib.axes._axes.Axes,
        df: pd.DataFrame,
        draw_legend: bool,
        add_annotation: bool,
    ):
        """
        Draw a default hist plot with a single distribution per available seeds

        Args:
            df (pd.DataFrame): Dataframe containing the data to be plotted
            ax (matplotlib.axes._axes.Axes): Axes object to plot on
            draw_legend (bool): Whether to draw the legend
            add_annotation (bool): Whether to add text annotation
        """
        df = df.sort_values(by="seed")
        n_colors = len(df.seed.unique())
        palette = sns.color_palette("crest", n_colors)

        for i, seed in enumerate(df.seed.unique()):
            _df = df[df.seed == seed]
            sns.kdeplot(
                data=_df,
                x="positions",
                fill=False,
                color=palette[i],
                bw_method=0.025,
                lw=1,
                label=seed,
                ax=ax,
            )
        # No legend for now, each line is a seed and it may get too messy
        # if draw_legend:
        #     ax.legend(
        #         loc="upper right",
        #         bbox_to_anchor=(1.25, 1),
        #         borderaxespad=0,
        #         title="Seed",
        #         prop={"size": 5},
        #     )
        # else:
        #     ax.legend().remove()

        if add_annotation:
            _add_text_labels(df, ax, x_pos=1.01, y_pos=0.25)

    def _split_effects_violin(
        ax: matplotlib.axes._axes.Axes,
        df: pd.DataFrame,
        draw_legend: bool,
        add_annotation: bool,
    ):
        """
        Draw a splitted violin by distinguishing mutations on individuals
        that drive the model score up vs down

        Args:
            df (pd.DataFrame): Dataframe containing the data to be plotted
            ax (matplotlib.axes._axes.Axes): Axes object to plot on
            draw_legend (bool): Whether to draw the legend
            add_annotation (bool): Whether to add text annotation
        """
        pal = sns.color_palette("deep")
        df["y"] = ""
        v = sns.violinplot(
            data=df,
            x="positions",
            y="y",
            hue="model_effect",
            hue_order=["Splicing Up", "Splicing Down"],
            split=True,
            bw_method=0.01,  # previous bw
            density_norm="count",
            common_norm=False,  # previous scale_hue
            orient="h",
            palette={"Splicing Up": pal[0], "Splicing Down": pal[1]},
            ax=ax,
        )
        if draw_legend:
            v.legend(
                loc="center right",
                bbox_to_anchor=(1.1, 1.1),
                fontsize="small",
                handleheight=0.5,
                handlelength=0.7,
                borderaxespad=0,
                frameon=False,
                title=None,
            )
        else:
            ax.legend().remove()

        if add_annotation:
            for i in range(2):
                if i == 0:
                    _df = df[df["model_diff"] >= 0]
                    _add_text_labels(_df, ax, x_pos=0.9, y_pos=0.8, split=True)
                else:
                    _df = df[df["model_diff"] < 0]
                    _add_text_labels(_df, ax, x_pos=0.9, y_pos=0.25, split=True)
        plt.subplots_adjust(left=0.05, right=0.9)

    assert any(
        not x for x in [split_effects, split_seeds]
    ), "Only one of split_effect or split_seeds can be True"

    df = df.copy()
    df["original_seq_id"] = seq_id
    df["model_diff"] = df.score - score
    df["model_effect"] = df.model_diff.apply(
        lambda x: "Splicing Up" if x >= 0 else "Splicing Down"
    )

    add_annotation = True
    if zoom_in:
        acc = ss_idx[1][0]
        don = ss_idx[1][1]
        df["positions"] = df.positions.apply(
            lambda x: [p for p in x if p >= acc - 100 and p <= don + 100]
        )

        df = df[df.positions.apply(lambda x: len(x) > 0)]
        seq_len = (don + 100) - (acc - 100)
        add_annotation = False

    df["total_n_perturb"] = df.n_perturbations.sum()
    df["total_n_perturb_split"] = df.groupby("model_effect")[
        "n_perturbations"
    ].transform("sum")

    df = df.explode("positions")
    df["positions"] = df.positions.astype(int)

    if split_seeds:
        _split_seeds_hist(
            ax, df, draw_legend=draw_legend, add_annotation=add_annotation
        )
        ax.set_ylabel("Distribution of positions")

    else:
        if split_effects:
            _split_effects_violin(
                ax, df, draw_legend=draw_legend, add_annotation=add_annotation
            )
            ax.set_ylabel("Distribution of positions")

        else:
            _default_hist(
                ax,
                df,
                seq_len=seq_len,
                zoom_in=zoom_in,
                draw_legend=draw_legend,
                add_annotation=add_annotation,
            )
            ax.set_ylabel("Number of mutations")

    if draw_exon_boundaries:
        for ss in ss_idx:
            ax.axvline(ss[0], color="grey", linestyle="--")
            ax.axvline(ss[1], color="grey", linestyle="--")

    if group_name:
        ax.text(
            0,
            1.15,
            f"Group:{group_name}",
            fontsize=10,
            ha="left",
            va="top",
            transform=ax.transAxes,
        )

    ax.set_xlabel("")

    if zoom_in:
        ax.set_xlim(acc - 100, don + 100)
