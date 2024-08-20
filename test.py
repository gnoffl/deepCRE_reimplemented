import pandas as pd
import unittest
import pyranges as pr


def compare_lists(list1, list2) -> bool:
    if len(list1) != len(list2):
        print(f"length of lists doesnt fit ({len(list1)}, {len(list2)})")
        return False
    
    list1 = sorted(list1)
    list2 = sorted(list2)
    for item1, item2 in zip(list1, list2):
        if item1 != item2:
            print(f"{item1} != {item2}")
            return False
    return True


class TestDeepCRE(unittest.TestCase):

    def test_compare_old_new(self):
        new = pd.read_csv("results/predictions/arabidopsis_deepcre_predict_240820_172417.csv", header=0)
        old = pd.read_csv("results/predictions/arabidopsis_deepcre_predict_240819_173613.csv", header=0)
        self.assertEqual(len(old), len(new))
        comparison = old.compare(new)
        self.assertTrue(comparison.empty)

    def test_gene_dist(self):
        annotation_path = "gene_models/Arabidopsis_thaliana.TAIR10.52.gtf"
        annot = pr.read_gtf(f=annotation_path, as_df=True)
        annot = annot[annot['gene_biotype'] == 'protein_coding']
        annot = annot[annot['Feature'] == 'gene']
        annot = annot[['Chromosome', 'Start', 'End', 'Strand', 'gene_id']]
        full_list = []
        list_longer_1000 = []
        for chrom, start, end, strand, gene_id in annot.values:
            if chrom in ["1", "2"]:
                full_list.append(gene_id)
                if end - start >= 1000:
                    list_longer_1000.append(gene_id)
        ignore_short = pd.read_csv("results/predictions/arabidopsis_deepcre_predict_240820_172417.csv", header=0)
        include_short = pd.read_csv("results/predictions/arabidopsis_deepcre_predict_240820_180424.csv", header=0)
        short_list = ignore_short["genes"].values
        long_list = include_short["genes"].values
        self.assertTrue(compare_lists(short_list, list_longer_1000))
        self.assertTrue(compare_lists(long_list, full_list))


if __name__ == "__main__":
    unittest.main()