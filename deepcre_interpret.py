import argparse
import os
from typing import List
import pandas as pd
from utils import extract_scores, get_time_stamp
import tensorflow as tf
import h5py


def save_results(output_name: str, shap_actual_scores, shap_hypothetical_scores, gene_ids_seqs: List, preds_seqs: List):
    file_name = os.path.splitext(os.path.basename(__file__))[0]
    with h5py.File(name=f'results/shap/{output_name}_{file_name}_{get_time_stamp()}.h5', mode='w') as h5_file:
        h5_file.create_dataset(name='contrib_scores', data=shap_actual_scores)
        h5_file.create_dataset(name="hypothetical_contrib_scores", data=shap_hypothetical_scores)
        pd.DataFrame({'gene_ids': gene_ids_seqs,
                    'preds': preds_seqs}).to_csv(path_or_buf=f'results/shap/{output_name}_shap_meta.csv', index=False)


def main():
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.disable_v2_behavior()
    tf.config.set_visible_devices([], 'GPU')

    parser = argparse.ArgumentParser(
                        prog='deepCRE',
                        description="""
                        This script performs the deepCRE prediction. We assume you have the following three directories:
                        tmp_counts (contains your counts files), genome (contains the genome fasta files),
                        gene_models (contains the gtf files)
                        """)

    parser.add_argument('--input',
                        help="""
                        This is a 5 column csv file with entries: genome, gtf, tpm, output name, number of chromosomes.""",
                        required=True)
    parser.add_argument('--model_case', help="Can be SSC or SSR", required=True)
    parser.add_argument('--ignore_small_genes', help="Ignore small genes, can be yes or no", required=True)

    args = parser.parse_args()
    data = pd.read_csv(args.input, sep=',', header=None,
                    dtype={0: str, 1: str, 2: str, 3: str, 4: int, 5: str},
                    names=['genome', 'gtf', 'tpm', 'output', 'counts'])
    print(data.head())
    if data.shape[1] != 5:
        raise Exception("Input file incorrect. Your input file must contain 5 columns and must be .csv")


    for genome, gtf, tpm_counts, output_name, num_chromosomes in data.values:
        results = extract_scores(genome=genome, annot=gtf, tpm_targets=tpm_counts, upstream=1000, downstream=500,
                    n_chromosome=num_chromosomes, ignore_small_genes=args.ignore_small_genes,
                    output_name=output_name, model_case=args.model_case)
        shap_actual_scores, shap_hypothetical_scores, one_hots_seqs, gene_ids_seqs, pred_seqs = results
        save_results(shap_actual_scores=shap_actual_scores, shap_hypothetical_scores=shap_hypothetical_scores,
                     output_name=output_name, gene_ids_seqs=gene_ids_seqs, preds_seqs=pred_seqs)


if __name__ == "__main__":
    main()