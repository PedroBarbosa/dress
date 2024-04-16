[![pytest](https://github.com/PedroBarbosa/dress/actions/workflows/run_tests.yml/badge.svg)](https://github.com/PedroBarbosa/dress/actions/workflows/run_tests.yml)

# DRESS - Deep learning-based Resource for Exploring Splicing Signatures

A toolkit for generating synthetic datasets related to RNA splicing.

## Running example

As for now, the package contains two commands:
 - `generate` to generate synthetic datasets from a start sequence.
 - `filter` to filter datasets by desired levels of splice site probability, PSI or dPSI.
 
To run an evolutionary search with exon 6 of FAS gene:

`dress generate data/examples/generate/raw_input/FAS_exon6/data.tsv`

To skip running the black box model (e.g, SpliceAI), run with `--dry_run`, which will return as fitness the proportional index (between 0 and 1) of the individual in the population.

The full list of argument options can be inspected with `dress generate --help` or by looking at the yaml configuration file at `dress/configs/generate.yaml`. 

The required transcript structure cache (from GENCODE v44) can be downloaded from [here](https://app.box.com/s/tbh293kqh1s9nbi624esl0c18maxuhss). Then, download the human genome hg38 (for example from [here](https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_45/GRCh38.primary_assembly.genome.fa.gz), uncompress it and optionally simplify chromosome headers with `sed '/^>/s/ .*//'`. Then, put both files in a single directory, which is given in `--cache_dir`. By default, it expects this data to be in `data/cache`.