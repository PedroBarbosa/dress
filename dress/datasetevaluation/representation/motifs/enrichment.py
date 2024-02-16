from abc import abstractmethod
import os
import pandas as pd
from dress.datasetgeneration.logger import setup_logger
from dress.datasetgeneration.os_utils import assign_proper_basename
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from matplotlib import pyplot as plt
import seaborn as sns


class MotifEnrichment:
    def __init__(self, df: pd.DataFrame, specific_basename: str = None, **kwargs):
        """
        Perform motif enrichment between two sequence groups

        Args:
            df (pd.DataFrame): Dataframe with sequence IDs (in rows)
        and motif counts (in columns). A `group` column must
        exist in the data defining the groups to be compared
            specific_basename (str): Specific basename to add to output.
        """

        if "logger" in kwargs:
            self.logger = kwargs["logger"]
        else:
            self.logger = setup_logger(level=0)

        if "group" not in df.columns:
            self.logger.warning(
                "Motif enrichment is performed between two sequence groups. To do so, two datasets must be given as input ('-d' and '-ad' options). Skipping."
            )
            return

        if df.group.nunique() != 2:
            self.logger.warning(
                f"Two different groups of sequences are required to perform motif enrichment. {df.group.nunique()} group(s) found: {','.join(df.group.unique())}. Skipping."
            )
            return

        for c in ["Score", "Delta_score"]:
            if c in df.columns:
                df.drop(columns=c, inplace=True)

        self.motif_counts = df
        self.outdir = os.path.join(kwargs.get("outdir", "output"), "motifs/enrichment")
        os.makedirs(self.outdir, exist_ok=True)
        self.outbasename = assign_proper_basename(kwargs.get("outbasename"))
        self.specific_basename = assign_proper_basename(specific_basename)

    @abstractmethod
    def visualize(self):
        ...


class FisherEnrichment(MotifEnrichment):
    def __init__(self, df: pd.DataFrame, specific_basename: str = None, **kwargs):
        super().__init__(df, specific_basename, **kwargs)

        if hasattr(self, "motif_counts"):
            self.results = self._run()

    def _run(self) -> pd.DataFrame:
        """
        Perform Fisher's exact test for motif enrichment between two groups

        Args:
            df (pd.DataFrame): Dataframe with motif counts

        Returns:
            pd.DataFrame: Dataframe with motif enrichment results.
        """
        df = self.motif_counts.copy()
        if "Seq_id" in df.columns:
            df.set_index("Seq_id", inplace=True)
        
        group_a_name = df.group.unique()[0]
        group_a = df[df.group == group_a_name].drop(columns="group")
        group_b_name = df.group.unique()[1]
        group_b = df[df.group == group_b_name].drop(columns="group")

        assert (
            len(group_a.columns) > 1
        ), "There must be at least two motifs in the data to test"

        out = []
        for RBP in group_a.columns:
            # Create a contingency table
            #        | GroupSeqs A | GroupSeqs B |
            #  RBP   |             |             |
            #  NoRBP |             |             |

            count_grp_a = group_a[RBP].sum()
            count_grp_b = group_b[RBP].sum()

            count_noRBP_grp_a = group_a.drop(columns=RBP).to_numpy().sum()
            count_noRBP_grp_b = group_b.drop(columns=RBP).to_numpy().sum()

            contingency_table = [
                [count_grp_a, count_grp_b],
                [count_noRBP_grp_a, count_noRBP_grp_b],
            ]

            try:
                odds_ratio, p_value = fisher_exact(
                    contingency_table, alternative="two-sided"
                )
                out.append([RBP, odds_ratio, p_value * len(group_a.columns)])

            except ValueError:
                self.logger.warning("Fisher's exact test failed for {}".format(RBP))
                continue
            self.logger.debug(
                "Test for {} motif: Odds_ratio: {}, p_value: {}".format(
                    RBP, odds_ratio, p_value
                )
            )

        # Sort the p-values in increasing order
        out = pd.DataFrame(out, columns=["RBP", "odds_ratio", "p_value"])
        out = out.sort_values(by="p_value")
        out = out.reset_index(drop=True)

        # Apply the Holm-Bonferroni correction
        reject, pvals_corrected, alphaSidak, alphaBonf = multipletests(
            out.p_value, alpha=0.05, method="holm"
        )

        # Add corrected p-values to the dataframe
        out["p_value_corrected"] = pvals_corrected
        out["reject"] = reject

        out.to_csv(
            os.path.join(
                self.outdir,
                f"{self.outbasename}{self.specific_basename}fisher_enrichment_{group_a_name}_vs_{group_b_name}.csv",
            ),
            index=False,
        )
        return out

    def visualize(self):
        """
        Visualize motif enrichment results

        Args:
            df (pd.DataFrame): Dataframe with motif enrichment results
        """
        if not hasattr(self, "results"):
            self.logger.warning("No motif enrichment results found for visualization.")
            return

        df = self.results.copy()

        # Sort by average corrected p-value
        df_avg = (
            df.groupby("RBP")["p_value_corrected"]
            .mean()
            .reset_index(name="avg_p_value_corrected")
        )
        df = df.merge(df_avg, on="RBP")
        df = df.sort_values("avg_p_value_corrected", ascending=True)

        # Plot just top 50 motifs

        if df["RBP"].nunique() > 50:
            self.logger.info("INFO", "Plotting just the top 50 motifs")

            top_50_motifs = df.RBP.drop_duplicates(keep="first").tolist()
            df = df[df.RBP.isin(top_50_motifs[:50])]

        _, ax = plt.subplots(figsize=(4, 7))

        # Create stripplot
        sns.stripplot(
            data=df,
            x="p_value_corrected",
            y="RBP",
            dodge=True,
            jitter=True,
            palette="Blues",
            size=6,
            alpha=0.7,
            hue=df["p_value_corrected"] < 0.05,
            hue_order=[True, False],
            legend=False,
            linewidth=1,
            edgecolor="black",
            clip_on=False,
            ax=ax,
        )

        ax.axvline(x=0.05, color="black", linestyle="--")
        ax.text(0.05, 1.01, "0.05", transform=ax.get_xaxis_transform(), fontsize=8)

        # ax.set_xlim([0, 0.2])
        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels(ax.get_yticklabels(), va="center", fontsize=8)
        ax.set_xlabel("Corrected p-value")
        ax.set_ylabel("")

        # handles, _ = ax.get_legend_handles_labels()
        # ax.legend(handles, ["Significant (p < 0.05)", "Non-significant"], loc="lower right")

        plt.savefig(
            os.path.join(
                self.outdir,
                f"{self.outbasename}{self.specific_basename}fisher_enrichment.pdf",
            ),
            bbox_inches="tight",
        )
        plt.close()
