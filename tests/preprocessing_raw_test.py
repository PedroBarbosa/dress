import gzip
import shutil
import pytest
from dress.datasetgeneration.os_utils import open_fasta
from dress.datasetgeneration.preprocessing.utils import tabular_file_to_genomics_df
from dress.datasetgeneration.preprocessing.gtf_cache import (
    extractGeneStructure,
    generate_pipeline_input,
)

raw_data = "tests/data/raw_data.tsv"
cache_dir = "tests/data"
genome_c = "tests/data/chr22.fa.gz"
genome = genome_c.replace(".gz", "")
level = 2

with gzip.open(genome_c, "rb") as f_in:
    with open(genome, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


class TestRawPreprocessing:
    @pytest.fixture
    def initial_setup(self):
        df = tabular_file_to_genomics_df(
            raw_data, col_index=0, is_0_based=False, header=0
        )
        extracted, absent_in_gtf = extractGeneStructure(
            df.as_df(), cache_dir=cache_dir, genome=genome, level=level
        )

        assert extracted.shape[0] == 8
        assert absent_in_gtf.shape[0] == 0
        return extracted, absent_in_gtf

    def test_full_sequence(self, initial_setup):
        extracted = initial_setup[0]

        data, _, na_exons = generate_pipeline_input(
            df=extracted,
            fasta=open_fasta(genome),
            extend_borders=100,
            use_full_seqs=True,
        )

        assert na_exons.shape[0] == 0
        len_test = {
            "chr22:29957036-29957088": 87889,
            "chr22:29970976-29971062": 21496,
            "chr22:36253838-36253991": 1922,
            "chr22:17812507-17812557": 6082,
            "chr22:31688156-31688227": 20416,
            "chr22:42601970-42602107": 3463,
            "chr22:43137081-43137298": 9041,
            "chr22:50457034-50457111": 2901,
        }

        len_test = {k: v + 200 for k, v in len_test.items()}

        header_test = {
            "chr22:29957036-29957088": "chr22:29883074-29971162(+)",
            "chr22:29970976-29971062": "chr22:29956936-29978631(+)",
            "chr22:36253838-36253991": "chr22:36252978-36255099(+)",
            "chr22:17812507-17812557": "chr22:17810603-17816884(-)",
            "chr22:31688156-31688227": "chr22:31681247-31701862(-)",
            "chr22:42601970-42602107": "chr22:42599598-42603260(-)",
            "chr22:43137081-43137298": "chr22:43132109-43141349(-)",
            "chr22:50457034-50457111": "chr22:50456392-50459492(-)",
        }

        acceptors_test = {
            "chr22:29957036-29957088": "100;73962;87902",
            "chr22:29970976-29971062": "100;14040;21506",
            "chr22:36253838-36253991": "100;860;1959",
            "chr22:17812507-17812557": "100;4327;6071",
            "chr22:31688156-31688227": "100;13635;16059",
            "chr22:42601970-42602107": "100;1153;3467",
            "chr22:43137081-43137298": "100;4051;7863",
            "chr22:50457034-50457111": "100;2381;2819",
        }

        donors_test = {
            "chr22:29957036-29957088": "285;74014;87988",
            "chr22:29970976-29971062": "152;14126;21595",
            "chr22:36253838-36253991": "241;1013;2021",
            "chr22:17812507-17812557": "194;4377;6181",
            "chr22:31688156-31688227": "206;13706;20515",
            "chr22:42601970-42602107": "490;1290;3562",
            "chr22:43137081-43137298": "187;4268;9140",
            "chr22:50457034-50457111": "237;2458;3000",
        }

        for _, seq_info in data.iterrows():
            exon = seq_info.exon
            header = seq_info.header
            acceptor_idx = seq_info.acceptor_idx
            donor_idx = seq_info.donor_idx
            seq = seq_info.seq

            # Assert that the header (after expanding exon coordinates) is correct
            assert header_test[exon] == header

            # Assert that sequences have the correct length
            assert len_test[exon] == len(seq)

            # Asser that sequences have the correct indexes for donors and acceptors
            assert acceptors_test[exon] == acceptor_idx
            assert donors_test[exon] == donor_idx

            # Assert that the acceptor and donor sites at the retrieved indexes are correct
            for i, (acceptor, donor) in enumerate(
                zip(acceptor_idx.split(";"), donor_idx.split(";"))
            ):
                if i == 1:
                    acceptor = int(acceptor)
                    donor = int(donor)

                    assert seq[acceptor - 2 : acceptor] == "AG"
                    assert seq[donor + 1 : donor + 3] == "GT"

    def test_trimmed_sequence(self, initial_setup):
        extracted = initial_setup[0]

        data, _, _ = generate_pipeline_input(
            df=extracted,
            fasta=open_fasta(genome),
            extend_borders=100,
            use_full_seqs=False,
        )

        len_test = {
            "chr22:29957036-29957088": 10053,
            "chr22:29970976-29971062": 10087,
            "chr22:36253838-36253991": 1922 + 200,
            "chr22:17812507-17812557": 6082 + 200,
            "chr22:31688156-31688227": 11981,
            "chr22:42601970-42602107": 3463 + 200,
            "chr22:43137081-43137298": 9041 + 200,
            "chr22:50457034-50457111": 2901 + 200,
        }

        header_test = {
            "chr22:29957036-29957088": "chr22:29952036-29962088(+)",
            "chr22:29970976-29971062": "chr22:29965976-29976062(+)",
            "chr22:36253838-36253991": "chr22:36252978-36255099(+)",
            "chr22:17812507-17812557": "chr22:17810603-17816884(-)",
            "chr22:31688156-31688227": "chr22:31681247-31693227(-)",
            "chr22:42601970-42602107": "chr22:42599598-42603260(-)",
            "chr22:43137081-43137298": "chr22:43132109-43141349(-)",
            "chr22:50457034-50457111": "chr22:50456392-50459492(-)",
        }

        acceptors_test = {
            "chr22:29957036-29957088": "<NA>;5000;<NA>",
            "chr22:29970976-29971062": "<NA>;5000;<NA>",
            "chr22:36253838-36253991": "100;860;1959",
            "chr22:17812507-17812557": "100;4327;6071",
            "chr22:31688156-31688227": "<NA>;5000;7424",
            "chr22:42601970-42602107": "100;1153;3467",
            "chr22:43137081-43137298": "100;4051;7863",
            "chr22:50457034-50457111": "100;2381;2819",
        }

        donors_test = {
            "chr22:29957036-29957088": "<NA>;5052;<NA>",
            "chr22:29970976-29971062": "<NA>;5086;<NA>",
            "chr22:36253838-36253991": "241;1013;2021",
            "chr22:17812507-17812557": "194;4377;6181",
            "chr22:31688156-31688227": "<NA>;5071;11880",
            "chr22:42601970-42602107": "490;1290;3562",
            "chr22:43137081-43137298": "187;4268;9140",
            "chr22:50457034-50457111": "237;2458;3000",
        }

        for _, seq_info in data.iterrows():
            exon = seq_info.exon

            header = seq_info.header
            acceptor_idx = seq_info.acceptor_idx
            donor_idx = seq_info.donor_idx
            seq = seq_info.seq

            # Assert that the header (after expanding exon coordinates) is correct
            assert header_test[exon] == header

            # Assert that sequences have the correct length
            assert len_test[exon] == len(seq)

            # Assert that sequences have the correct indexes for donors and acceptors
            assert acceptors_test[exon] == acceptor_idx
            assert donors_test[exon] == donor_idx

            # Assert that the acceptor and donor sites at the retrieved indexes are correct
            for i, (acceptor, donor) in enumerate(
                zip(acceptor_idx.split(";"), donor_idx.split(";"))
            ):
                # Case where just the upstream is trimmed
                if exon == "chr22:31688156-31688227":
                    if i == 0:
                        assert acceptor == "<NA>"
                        assert donor == "<NA>"

                    if i == 2:
                        acceptor = int(acceptor)
                        donor = int(donor)
                        assert seq[acceptor - 2 : acceptor] == "AG"
                        assert seq[donor + 1 : donor + 3] == "TA"
                        assert seq[acceptor - 8 : acceptor + 4] == "ATCCTTAGGTGT"
                        assert seq[donor - 1 : donor + 5] == "AATAAC"
                        assert len(seq[acceptor : donor + 1]) == 4457

                if i == 1:
                    acceptor = int(acceptor)
                    donor = int(donor)

                    assert seq[acceptor - 2 : acceptor] == "AG"
                    assert seq[donor + 1 : donor + 3] == "GT"
