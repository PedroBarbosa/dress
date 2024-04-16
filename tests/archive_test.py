from dress.datasetgeneration.dataset import Dataset
from dress.datasetgeneration.archive import Archive
import numpy as np
import os 
dataset = Dataset(f"{os.path.dirname(os.path.abspath(__file__))}/data/dataset.csv.gz"
)

class TestFromGeneratedDataset:
    archive = Archive(diversity_metric="normalized_shannon", dataset=dataset.data.iloc[1:])

    def test_archive_size(self):
        assert len(self.archive) == 4983

    def test_archive_diversity(self):
        assert np.isclose(self.archive.diversity(), 0.99968626, atol=1e-05)

    def test_archive_intra_bin_diversity(self):
        diversity_per_bin = self.archive.diversity_per_bin().values()
        avg_diversity_per_bin = np.mean(
            list(filter(lambda x: x != "--", diversity_per_bin))
        )
        assert np.isclose(avg_diversity_per_bin, 0.94, atol=1e-02)

    def test_archive_quality(self):
        assert np.isclose(self.archive.quality, 0.9881, atol=1e-05)

    def test_other_metrics(self):
        metrics = self.archive.metrics
        assert metrics["Empty_bin_ratio"] == 0
        assert metrics["Low_count_bin_ratio"] == 0
        assert metrics["Avg_number_diff_units"] == 4.6903
        assert metrics["Avg_edit_distance"] == 10.0281

    def test_archive_slicing(self):
        assert len(self.archive[:]) == len(self.archive)
        assert len(self.archive[:0.1]) == 497
        assert len(self.archive[0:0.1]) == 497
        assert len(self.archive[0.1:]) == 4486
        assert len(self.archive[0.8:]) == 975
        assert len(self.archive[0.4:0.5]) == 508
