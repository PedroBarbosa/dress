import argparse
import itertools
from itertools import combinations
import math
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def generate_deletions(sequence: str, max_size: int):
    deletions = []

    for size in range(1, max_size + 1):
        for pos in range(len(sequence)):
            if pos + size > len(sequence):
                break

            deletions.append(sequence[pos : pos + size])

    return deletions


def generate_insertions(sequence: str, max_size: int):
    """
    Does not allow to generate insertion at the beginning and end of the sequence
    """
    nucleotides = ["A", "T", "C", "G"]
    insertions = []

    for size in range(1, max_size + 1):
        for _ in range(1, len(sequence)):
            for ins in itertools.product(nucleotides, repeat=size):
                insertions.append("".join(ins))

    return insertions


def generate_mnps(sequence: str, max_size: int):
    nucleotides = ["A", "T", "C", "G"]
    mnps = []

    for size in range(2, max_size + 1):
  
        for pos in range(len(sequence) - size + 1):
            start_pos = pos
            end_pos = pos + size
            seq = sequence[start_pos:end_pos]

            for changes in itertools.product(nucleotides, repeat=size):
                if any([c == seq[i] for i, c in enumerate(changes)]):
                    continue

                mnp = "".join(changes)
                mnps.append(mnp)

    return mnps


def generate(grammar_option: str, seq: str, **kwargs):
    # assert len(seq) < 20, "Sequence too long to test"

    nucleotides = ["A", "T", "C", "G"]
    out = []

    if grammar_option == "SNV":
        for i in range(len(seq)):
            for n in nucleotides:
                if n != seq[i]:
                    snp = seq[:i] + n + seq[i + 1 :]
                    out.append(snp)

    elif grammar_option == "MNP":
        out = generate_mnps(seq, max_size=kwargs["max_mnp_size"])

    elif grammar_option == "INS":
        out = generate_insertions(seq, max_size=kwargs["max_insertion_size"])

    elif grammar_option == "DEL":
        out = generate_deletions(seq, max_size=kwargs["max_deletion_size"])

    return len(out)


def n_possible_of_given_type(grammar_option: str, seq_size: int, **kwargs) -> int:
    """
    For a given seq size, calculates the possible
    number of single mutations to do of given type `grammar_option`
    """

    if grammar_option == "SNV":
        n_seqs = seq_size * 3

    elif grammar_option == "MNP":
        max_mnp_size = kwargs["max_mnp_size"]
        n_seqs = []

        for _mnp_size in range(2, max_mnp_size + 1):
            n_slides = seq_size - _mnp_size + 1
            n_seqs.append(n_slides * (3**_mnp_size))

        n_seqs = sum(n_seqs)

    elif grammar_option == "INS":
        max_ins_size = kwargs["max_insertion_size"]
        n_seqs = []

        for _mnp_size in range(1, max_ins_size + 1):
            n_slides = seq_size - 1
            n_seqs.append(n_slides * (4**_mnp_size))

        n_seqs = sum(n_seqs)

    elif grammar_option == "DEL":
        max_del_size = kwargs["max_deletion_size"]
        n_seqs = []
        for _del_size in range(1, max_del_size + 1):
            if _del_size > seq_size:
                break

            if _del_size == 1:
                n_seqs.append(seq_size)

            else:
                n_seqs.append(seq_size - _del_size + 1)

        n_seqs = sum(n_seqs)

    return n_seqs

def line_plot(df: pd.DataFrame, max_number_of_changes: int):
    df["Number of mutations applied"] = df["Number of mutations applied"].astype(str)
    cat_order = [str(i) for i in range(1, max_number_of_changes + 1)]
    df["Number of mutations applied"] = pd.Categorical(
            df["Number of mutations applied"], categories=cat_order, ordered=True
        )
    ax = sns.lineplot(
        data=df, x="Number of mutations applied", y="Number of sequences", palette='mako', hue="Grammar"
    )
    ax.set_xticklabels(cat_order)
    ax.set_xticks(range(len(cat_order)))
    ax.set_ylabel("Number of sequences (log scale)")

    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("complexity_by_grammar_and_N_mutations.pdf")
    plt.close()
    
def main():
    parser = argparse.ArgumentParser(
        description="Calculate number of possible combinations and expected runtime for a given "
        "sequence size and grammar complexity."
    )

    parser.add_argument(
        "--seq_size",
        type=int,
        default=10120,
        help="Input sequence size that accounts for model resolution (10kb nucleotides). "
        "Default: 10120, accounting for an exon size of 120bp",
    )
    parser.add_argument(
        "--max_number_of_changes",
        type=int,
        default=5,
        help="Max number of changes in a sequence to "
        "account for. Calculation will be made up to this number. Default: 5",
    )
    parser.add_argument(
        "--possible_changes",
        nargs="+",
        choices=["SNV", "MNP", "INS", "DEL"],
        default=["SNV", "MNP", "INS", "DEL"],
        help="Possible changes in sequences to account for in the grammar design. Default: ['SNV', 'MNP', 'INS', 'DEL']",
    )
    parser.add_argument(
        "--max_mnp_size",
        type=int,
        default=2,
        help="max MNP size to consider. Default: 2",
    )

    parser.add_argument(
        "--max_deletion_size", type=int, default=5, help="max DEL size. Default: 5"
    )

    parser.add_argument(
        "--max_insertion_size", type=int, default=5, help="max INS size. Default: 5"
    )

    parser.add_argument(
        "--spliceai_time_per_batch",
        type=float,
        default=0.4,
        help="Expected time (in seconds) a batch of 64 "
        "sequences of size `seq_size` takes to run on a GPU. Default: 0.4, for a sequence of size 10120.",
    )

    args = parser.parse_args()

    kwargs = {
        "max_mnp_size": args.max_mnp_size,
        "max_deletion_size": args.max_deletion_size,
        "max_insertion_size": args.max_insertion_size,
    }

    assert (
        args.seq_size > 10000
    ), "Sequence size must be larger than 10000 to account for model resolution."

    # Checks
    seqs = ["AT", "ATA", "ATAT", "ATATG", "ATATGC", "ATATGCC", "ATATGCCG"]

    for ty in ["SNV", "MNP", "INS", "DEL"]:

        for s in seqs:
            obs = generate(grammar_option=ty, seq=s, **kwargs)
            calc = n_possible_of_given_type(
                seq_size=len(s),
                grammar_option=ty,
                **kwargs,
            )
            assert obs == calc, f"Number of generated sequences ({ty}={obs}) does not match calculated values: {calc}"
    
    # Calculate number of possible unique sequences
    # after applying 1 mutation of given type
    n_sequences_per_type = {}
    for ty in args.possible_changes:
        n_sequences_per_type[ty] = n_possible_of_given_type(
            grammar_option=ty,
            seq_size=args.seq_size,
            **kwargs,
        )

    # Generate all possible grammar combinations,
    # considering the allowed mutation types provided
    grammar_combs, out = [], []
    for i in range(1, len(args.possible_changes) + 1):
        c = [list(x) for x in list(combinations(args.possible_changes, i))]
        grammar_combs.extend(c)

    # For each grammar
    for grammar in grammar_combs:
        name = "_".join(grammar)
        n_possible_changes = 0

        # Sum the total number of possible unique
        # sequences after applying 1 single mutation
        for mutation_type in grammar:
            n_possible_changes += n_sequences_per_type[mutation_type]

        # Calculate the total number of possible unique
        # sequences up to the max number of changes allowed
        for m in range(1, args.max_number_of_changes + 1):
            total = n_possible_changes**m
            out.append([name, m, total])

    df = pd.DataFrame(
        out, columns=["Grammar", "Number of mutations applied", "Number of sequences"]
    )

    if len(args.possible_changes) == 4:
        cat_order = [
            "SNV",
            "MNP",
            "DEL",
            "INS",
            "SNV_MNP",
            "SNV_DEL",
            "SNV_INS",
            "MNP_DEL",
            "MNP_INS",
            "INS_DEL",
            "SNV_MNP_DEL",
            "SNV_MNP_INS",
            "SNV_INS_DEL",
            "MNP_INS_DEL",
            "SNV_MNP_INS_DEL",
        ]
        df["Grammar"] = pd.Categorical(df.Grammar, categories=cat_order, ordered=True)

        df = df.sort_values("Grammar")
    line_plot(df, args.max_number_of_changes)
    
    df_wider = df.pivot_table(
        index="Grammar",
        columns="Number of mutations applied",
        values="Number of sequences",
    )

    print("Number of sequences:")
    print(df_wider)
    # By days
    df_wider = df_wider.apply(lambda x: ((x / 64) * args.spliceai_time_per_batch) / 86400)
    # By minutes
    #df_wider = df_wider.apply(lambda x: ((x / 64) * args.spliceai_time_per_batch) / 60)
    print("Expected time to run (days):")
    print(df_wider)


if __name__ == "__main__":
    main()
