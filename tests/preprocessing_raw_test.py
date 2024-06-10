import gzip
import os
import shutil
import pytest
import tempfile
from dress.datasetgeneration.os_utils import open_fasta
from dress.datasetgeneration.preprocessing.utils import tabular_file_to_genomics_df
from dress.datasetgeneration.preprocessing.gtf_cache import (
    extractGeneStructure,
    generate_pipeline_input,
)


# abs_path = os.path.dirname(os.path.abspath(__file__))
# raw_data = os.path.join(abs_path,"data/raw_data.tsv")
# cache_dir = os.path.join(abs_path,"data")
# genome_c =  os.path.join(abs_path, "data/chr22.fa.gz")
# level = 2

# with gzip.open(genome_c, "rb") as f_in:
#     tmp_f = tempfile.NamedTemporaryFile()
#     genome = tmp_f.name
#     with open(genome, "wb") as f_out:
#         shutil.copyfileobj(f_in, f_out)

@pytest.fixture(scope="module")
def setup_paths():
    abs_path = os.path.dirname(os.path.abspath(__file__))
    raw_data = os.path.join(abs_path, "data/raw_data.tsv")
    cache_dir = os.path.join(abs_path, "data")
    genome_c = os.path.join(abs_path, "data/chr22.fa.gz")

    with gzip.open(genome_c, "rb") as f_in:
        tmp_f = tempfile.NamedTemporaryFile(delete=False)
        genome = tmp_f.name
        with open(genome, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    yield raw_data, cache_dir, genome

    os.remove(genome)

@pytest.fixture
def initial_setup(setup_paths):
    raw_data, cache_dir, genome = setup_paths
    df = tabular_file_to_genomics_df(
        raw_data, col_index=0, is_0_based=False, header=0
    )
    extracted, absent_in_gtf = extractGeneStructure(
        df.as_df(), cache_dir=cache_dir, genome=genome, level=2
    )

    assert extracted.shape[0] == 8
    assert absent_in_gtf.shape[0] == 0
    return extracted, absent_in_gtf

@pytest.fixture
def common_tests():
    def assert_common_tests(seq_info, len_test, header_test, acceptors_test, donors_test):
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
            if i == 1:
                acceptor = int(acceptor)
                donor = int(donor)
                assert seq[acceptor - 2 : acceptor] == "AG"
                assert seq[donor + 1 : donor + 3] == "GT"

    return assert_common_tests

class TestRawPreprocessing:
    
    def test_full_sequence(self, setup_paths, initial_setup, common_tests):
        extracted, _ = initial_setup
        _, cache_dir, genome = setup_paths

        data, _, na_exons = generate_pipeline_input(
            df=extracted,
            fasta=open_fasta(genome, cache_dir),
            extend_borders=100,
            use_full_triplet=True,
            use_model_resolution=False,
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
            common_tests(seq_info, len_test, header_test, acceptors_test, donors_test)

    def test_trimmed_sequence(self, setup_paths, initial_setup, common_tests):
        extracted, _ = initial_setup
        _, cache_dir, genome = setup_paths

        data, _, _ = generate_pipeline_input(
            df=extracted,
            fasta=open_fasta(genome, cache_dir),
            extend_borders=100,
            use_full_triplet=False,
            use_model_resolution=False,
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
            common_tests(seq_info, len_test, header_test, acceptors_test, donors_test)

            # Specific tests for trimmed (default) resolution
            exon = seq_info.exon
            acceptor_idx = seq_info.acceptor_idx
            donor_idx = seq_info.donor_idx
            seq = seq_info.seq

            # Case where just the upstream is trimmed
            if exon == "chr22:31688156-31688227":      
                for i, (acceptor, donor) in enumerate(zip(acceptor_idx.split(";"), donor_idx.split(";"))):
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

    def test_model_resolution(self, setup_paths, initial_setup, common_tests):
        extracted, _ = initial_setup
        _, cache_dir, genome = setup_paths

        data, _, _ = generate_pipeline_input(
            df=extracted,
            fasta=open_fasta(genome, cache_dir),
            extend_borders=100,
            use_full_triplet=False,
            use_model_resolution=True,
            model="spliceai"
        )

        len_test = {
            "chr22:29957036-29957088": 53 + 10000,
            "chr22:29970976-29971062": 87 + 10000,
            "chr22:36253838-36253991": 154 + 10000,
            "chr22:17812507-17812557": 51 + 10000,
            "chr22:31688156-31688227": 72 + 10000,
            "chr22:42601970-42602107": 138 + 10000,
            "chr22:43137081-43137298": 218 + 10000,
            "chr22:50457034-50457111": 78 + 10000,
        }

        header_test = {
            "chr22:29957036-29957088": "chr22:29952036-29962088(+)",
            "chr22:29970976-29971062": "chr22:29965976-29976062(+)",
            "chr22:36253838-36253991": "chr22:36248838-36258991(+)",
            "chr22:17812507-17812557": "chr22:17807507-17817557(-)",
            "chr22:31688156-31688227": "chr22:31683156-31693227(-)",
            "chr22:42601970-42602107": "chr22:42596970-42607107(-)",
            "chr22:43137081-43137298": "chr22:43132081-43142298(-)",
            "chr22:50457034-50457111": "chr22:50452034-50462111(-)",
        }

        acceptors_test = {
            "chr22:29957036-29957088": "<NA>;5000;<NA>",
            "chr22:29970976-29971062": "<NA>;5000;<NA>",
            "chr22:36253838-36253991": "4240;5000;6099",
            "chr22:17812507-17812557": "773;5000;6744",
            "chr22:31688156-31688227": "<NA>;5000;7424",
            "chr22:42601970-42602107": "3947;5000;7314",
            "chr22:43137081-43137298": "1049;5000;8812",
            "chr22:50457034-50457111": "2719;5000;5438",
        }

        donors_test = {
            "chr22:29957036-29957088": "<NA>;5052;<NA>",
            "chr22:29970976-29971062": "<NA>;5086;<NA>",
            "chr22:36253838-36253991": "4381;5153;6161",
            "chr22:17812507-17812557": "867;5050;6854",
            "chr22:31688156-31688227": "<NA>;5071;<NA>",
            "chr22:42601970-42602107": "4337;5137;7409",
            "chr22:43137081-43137298": "1136;5217;10089",
            "chr22:50457034-50457111": "2856;5077;5619",
        }

        for _, seq_info in data.iterrows():
            common_tests(seq_info, len_test, header_test, acceptors_test, donors_test)

            exon = seq_info.exon
            acceptor_idx = seq_info.acceptor_idx
            donor_idx = seq_info.donor_idx
            seq = seq_info.seq

            if exon == "chr22:31688156-31688227":
                for i, (acceptor, donor) in enumerate(
                    zip(acceptor_idx.split(";"), donor_idx.split(";"))
                ):
                    if i == 0:
                        assert acceptor == "<NA>"
                        assert donor == "<NA>"

                    if i == 2:
                        acceptor = int(acceptor)
                        assert seq[acceptor - 2 : acceptor] == "AG"
                        assert seq[acceptor - 8 : acceptor + 4] == "ATCCTTAGGTGT"
                        
                        assert donor == "<NA>"
                        assert len(seq[acceptor:]) == 2648