import os
import re
import pandas as pd
import unittest
import pyranges as pr

from utils import make_absolute_path
from deepcre_predict import find_newest_model_path

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
    

def test_compare_old_new():
    new = pd.read_csv("results/predictions/arabidopsis_deepcre_predict_240820_172417.csv", header=0)
    old = pd.read_csv("results/predictions/arabidopsis_deepcre_predict_240819_173613.csv", header=0)
    if not len(old) == len(new):
        raise ValueError("legths of compared dataframes arent equal!")
    comparison = old.compare(new)
    if not comparison.empty:
        print(comparison)
        raise ValueError("DFs arent the same!")
    

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
    if not compare_lists(short_list, list_longer_1000):
        raise ValueError("short lists arent equal!")
    if not compare_lists(long_list, full_list):
        raise ValueError("full lists arent equal!")


def test_regex():
    # regex_string = "^arabidopsis_(\\d+)_SSR_train_ssr_models_\\d+_\\d+\\.h5&"
    regex_string = "^arabidopsis_1_SSR_train_ssr_models_\d+_\d+\.h5$"
    regex = re.compile(regex_string)
    match = regex.match("arabidopsis_1_SSR_train_ssr_models_240816_183905.h5")
    if match: 
        print(match.group(0))
        print(match.group(1))
    else:
        print("no match found")


class TestDeepCRE(unittest.TestCase):
    
    def test_model_finding(self):
        results = find_newest_model_path(output_name="arabidopsis", model_case="SSR", model_path="test_folder/model_names")
        for key, value in results.items():
            self.assertTrue(key in ["1", "2"])
        path_to_models = make_absolute_path("test_folder", "model_names", start_file=__file__)
        self.assertEqual(results["1"], os.path.join(path_to_models, "arabidopsis_1_SSR_train_ssr_models_240822_103323.h5"))
        self.assertEqual(results["2"], os.path.join(path_to_models, "arabidopsis_2_SSR_train_ssr_models_240822_105523.h5"))
        


if __name__ == "__main__":
    unittest.main()
    # test_regex()