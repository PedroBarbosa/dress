import os
import sqlite3
import subprocess
import time

import pandas as pd


def create_db(data: pd.DataFrame | str, **args):

    logger = args["logger"]
    logger.debug("Creating database.")
    outdir = f"{args['outdir']}/motifs"

    conn = sqlite3.connect(f"{outdir}/motifs.db")
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS motifs;")
    cursor.execute(
        "CREATE TABLE motifs (id INTEGER, \
                    Seq_id TEXT, \
                    RBP_name TEXT, \
                    RBP_motif TEXT, \
                    Start INTEGER, \
                    End INTEGER, \
                    RBP_name_motif TEXT, \
                    Has_self_submotif BOOL, \
                    Has_other_submotif BOOL, \
                    Is_high_density_region BOOL, \
                    N_at_density_block SMALLINT, \
                    distance_to_cassette_acceptor MEDIUMINT, \
                    distance_to_cassette_donor MEDIUMIN, \
                    is_in_exon BOOL, \
                    location ENUM[Intron_upstream_2, Intron_upstream, Intron_downstream, Intron_downstream_2, \
                    Exon_upstream_fully_contained, Exon_upstream_acceptor_region, Exon_upstream_donor_region, \
                    Exon_cassette_fully_contained, Exon_cassette_acceptor_region, Exon_cassette_donor_region, \
                    Exon_downstream_fully_contained, Exon_downstream_acceptor_region, Exon_downstream_donor_region], \
                    distance_to_acceptor INTEGER, \
                    distance_to_donor INTEGER, \
                    PRIMARY KEY(id, RBP_name, location, Start, End)) WITHOUT ROWID;"
    )

    if isinstance(data, pd.DataFrame):
        data.to_sql("motifs", conn, if_exists="replace", index=False)

    elif isinstance(data, str):
        assert os.path.isfile(data), f"File {data} does not exist."
        if data.endswith(".gz"):
            p1 = subprocess.Popen(["gunzip", "-c", data], stdout=subprocess.PIPE)
            subprocess.run(
                [
                    "sqlite3",
                    "motifs.db",
                    "-cmd",
                    ".mode tabs",
                    ".import --skip 1 '|cat -' motifs",
                ],
                stdin=p1.stdout,
                capture_output=True,
            )

        else:
            subprocess.run(
                [
                    "sqlite3",
                    "motifs.db",
                    "-cmd",
                    ".mode tabs",
                    ".import --skip 1 " + data + " motifs",
                ],
                capture_output=True,
            )

    logger.info("Database successfully created.")
    return cursor
