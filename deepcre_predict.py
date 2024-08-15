import argparse
import os
from pyfaidx import Fasta
import numpy as np
from tensorflow.keras.models import load_model #type:ignore
import pyranges as pr

import pandas as pd
from utils import get_filename_from_path, get_time_stamp, one_hot_encode


def predict(genome, annot, tpm_targets, upstream, downstream, val_chromosome, ignore_small_genes,
            output_name, model_case):
    genome = Fasta(filename=f"genome/{genome}", as_raw=True, read_ahead=10000, sequence_always_upper=True)
    tpms = pd.read_csv(filepath_or_buffer=f"tpm_counts/{tpm_targets}", sep=',')
    tpms.set_index('gene_id', inplace=True)
    annot = pr.read_gtf(f=f"gene_models/{annot}", as_df=True)
    annot = annot[annot['gene_biotype'] == 'protein_coding']
    annot = annot[annot['Feature'] == 'gene']
    annot = annot[['Chromosome', 'Start', 'End', 'Strand', 'gene_id']]
    annot = annot[annot['Chromosome'] == val_chromosome]
    expected_final_size = 2 * (upstream + downstream) + 20

    x, y, gene_ids = [], [], []
    for chrom, start, end, strand, gene_id in annot.values:#type:ignore
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
            x.append(seq)
            y.append(tpms.loc[gene_id, 'target'])
            gene_ids.append(gene_id)

    x, y, gene_ids = np.array(x), np.array(y), np.array(gene_ids)

    # Masking
    x[:, upstream:upstream + 3, :] = 0
    x[:, upstream + (downstream * 2) + 17:upstream + (downstream * 2) + 20, :] = 0

    model = load_model(f"saved_models/{model_case}_{output_name}_model_{val_chromosome}.h5")
    pred_probs = model.predict(x).ravel()
    return x, y, pred_probs, gene_ids, model


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

    folder_name = os.path.join('results', 'predictions')
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    file_name = get_filename_from_path(__file__)


    for genome, gtf, tpm_counts, output_name, num_chromosomes in data.values:
        true_targets, preds, genes = [], [], []
        for chrom in range(1, num_chromosomes + 1):
            _, y, pred_probs, gene_ids, _ = predict(genome=genome, annot=gtf, tpm_targets=tpm_counts, upstream=1000,
                                                    downstream=500, val_chromosome=str(chrom), output_name=output_name,
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