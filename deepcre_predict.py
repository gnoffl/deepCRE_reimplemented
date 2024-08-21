import argparse
import os
from typing import Dict, List, Tuple
import numpy as np
from tensorflow.keras.models import load_model #type:ignore

import pandas as pd
from utils import get_filename_from_path, get_time_stamp, load_input_files, one_hot_encode, make_absolute_path


def find_newest_model_path(output_name: str, val_chromosome: str, model_case: str) -> str:
    """finds path to newest model fitting the given parameters

    Args:
        output_name (str): output name the was used for model training
        val_chromosome (str): validation chromosome of the model
        model_case (str): SSR or SSC for the model to be loaded

    Raises:
        ValueError: raises an error if no fitting model is found

    Returns:
        str: path to the newest model fitting the given parameters
    """
    path_to_models = make_absolute_path("saved_models", start_file=__file__)
    candidate_models = [model for model in os.listdir(path_to_models) if model.startswith(f"{output_name}_{val_chromosome}_{model_case}") and model.endswith(".h5")]
    if not candidate_models:
        raise ValueError("no trained models fitting the given parameters were found! Consider training models first (train_ssr_models.py)")
    candidate_models.sort()
    path_to_newest_model = os.path.join(path_to_models, candidate_models[-1])
    return path_to_newest_model


def extract_genes(genome, annot, upstream, downstream, ignore_small_genes, tpms, target_chromosomes: Tuple[str, ...]) -> Dict[str, Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]]:
    extracted_seqs = {}
    expected_final_size = 2 * (upstream + downstream) + 20
    for chrom, start, end, strand, gene_id in annot.values:#type:ignore
        # skip all chromosomes that are not in the target chromosomes. Empty tuple () means, that all chromosomes should be extracted
        if target_chromosomes != () and chrom not in target_chromosomes:
            continue

        gene_size = end - start
        extractable_downstream = downstream if gene_size // 2 > downstream else gene_size // 2
        prom_start, prom_end = start - upstream, start + extractable_downstream
        term_start, term_end = end - extractable_downstream, end + upstream

        promoter = one_hot_encode(genome[chrom][prom_start:prom_end])
        terminator = one_hot_encode(genome[chrom][term_start:term_end])
        extracted_size = promoter.shape[0] + terminator.shape[0]
        central_pad_size = expected_final_size - extracted_size

        pad_size = 20 if ignore_small_genes.lower() == 'yes' else central_pad_size

        if strand == '+':
            seq = np.concatenate([
                promoter,
                np.zeros(shape=(pad_size, 4)),
                terminator
            ])
        else:
            seq = np.concatenate([
                terminator[::-1],
                np.zeros(shape=(pad_size, 4)),
                promoter[::-1]
            ])

        if seq.shape[0] == expected_final_size:
            extracted_tuple = extracted_seqs.get(chrom, ())
            if extracted_tuple == ():
                x, y, gene_ids = [], [], []
            else:
                x = extracted_tuple[0]
                y = extracted_tuple[1]
                gene_ids = extracted_tuple[2]
            # print(x)
            # print(type(x))
            x.append(seq)
            y.append(tpms.loc[gene_id, 'target'])
            gene_ids.append(gene_id)
            extracted_seqs[chrom] = (x, y, gene_ids)

    for chrom, tuple_ in extracted_seqs.items():
        x, y, gene_ids = tuple_
        x, y, gene_ids = np.array(x), np.array(y), np.array(gene_ids)
        extracted_seqs[chrom] = (x, y, gene_ids)
    return extracted_seqs


def predict_self(genome, annotation, tpm_targets, extragenic, intragenic, val_chromosome, ignore_small_genes,
            output_name, model_case):

    extracted_genes = extract_genes(genome, annotation, extragenic, intragenic, ignore_small_genes, tpm_targets, target_chromosomes=())
    print(extracted_genes.keys())
    x, y, gene_ids = extracted_genes[str(val_chromosome)]

    # Masking
    x[:, extragenic:extragenic + 3, :] = 0
    x[:, extragenic + (intragenic * 2) + 17:extragenic + (intragenic * 2) + 20, :] = 0

    newest_model_path = find_newest_model_path(output_name=output_name, val_chromosome=val_chromosome, model_case=model_case)
    model = load_model(newest_model_path)
    pred_probs = model.predict(x).ravel()
    return x, y, pred_probs, gene_ids, model

def predict_other():
    pass


def parse_args():
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
    return args


def main():
    args = parse_args()
    data = pd.read_csv(args.input, sep=',', header=None,
                    dtype={0: str, 1: str, 2: str, 3: str, 4: int, 5: str},
                    names=['genome', 'gtf', 'tpm', 'output', 'counts'])
    print(data.head())
    if data.shape[1] != 5:
        raise Exception("Input file incorrect. Your input file must contain 5 columns and must be .csv")

    folder_name = make_absolute_path('results', 'predictions', start_file=__file__)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    file_name = get_filename_from_path(__file__)


    for genome_file_name, annotation_file_name, tpm_counts_file_name, output_name, num_chromosomes in data.values:
        true_targets, preds, genes = [], [], []
        loaded_input_files = load_input_files(genome_file_name=genome_file_name, annotation_file_name=annotation_file_name, tpm_counts_file_name=tpm_counts_file_name)
        genome = loaded_input_files["genome"]
        annotation = loaded_input_files["annotation"]
        tpms = loaded_input_files["tpms"]
        for chrom in range(1, num_chromosomes + 1):
            _, y, pred_probs, gene_ids, _ = predict_self(genome=genome, annotation=annotation, tpm_targets=tpms, extragenic=1000,
                                                    intragenic=500, val_chromosome=str(chrom), output_name=output_name,
                                                    model_case=args.model_case, ignore_small_genes=args.ignore_small_genes)
            true_targets.extend(y)
            preds.extend(pred_probs)
            genes.extend(gene_ids)

        result = pd.DataFrame({'true_targets': true_targets, 'pred_probs': preds, 'genes': genes})
        print(result.head())
        output_location = os.path.join(folder_name, f'{output_name}_{file_name}_{get_time_stamp()}.csv')
        result.to_csv(output_location, index=False)

if __name__ == "__main__":
    main()