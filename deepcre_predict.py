import argparse
import os
from typing import Any, Dict, List, Tuple
import numpy as np
from tensorflow.keras.models import load_model #type:ignore
import pandas as pd
import re

from utils import get_filename_from_path, get_time_stamp, load_input_files, one_hot_encode, make_absolute_path
from train_ssr_models import extract_genes


def find_newest_model_path(output_name: str, model_case: str, val_chromosome: str = "", model_path: str = "") -> Dict[str, str]:
    """finds path to newest model fitting the given parameters

    Args:
        output_name (str): output name the was used for model training
        val_chromosome (str): validation chromosome of the model. If it is not given, all models regardless of the val_chromosome will be returned
        model_case (str): SSR or SSC for the model to be loaded
        model_path (str): path to the directory where models are stored. used for testing, probably not really stable

    Raises:
        ValueError: raises an error if no fitting model is found

    Returns:
        List[str]: List of path to the newest model fitting the given parameters for a single chromosome, or all fitting models if chromosome is ommitted.
    """
    if model_path == "":
        path_to_models = make_absolute_path("saved_models", start_file=__file__)
    else:
        path_to_models = make_absolute_path(model_path, start_file=__file__)
    # ^ and $ mark start and end of a string. \d singnifies any digit. \d+ means a sequence of digits with at least length 1
    # more detailed explanation at https://regex101.com/, put in "^ara_(\d+)_ssr_\d+_\d+\.h5$"
    if val_chromosome == "":
        regex_string = f"^{output_name}_(.+)_{model_case}_train_ssr_models_\d+_\d+\.h5$"                                                                    #type:ignore
    else:
        regex_string = f"^{output_name}_{val_chromosome}_{model_case}_train_ssr_models_\d+_\d+\.h5$"                                                        #type:ignore
    regex = re.compile(regex_string)
    candidate_models = [model for model in os.listdir(path_to_models)]
    fitting_models = {}
    for candidate in candidate_models:
        match = regex.match(candidate)
        if match:
            # group 1 is the "(.+)" part of the regex, so the place where the chromosome is located in the file name
            chromosome = val_chromosome if val_chromosome else match.group(1)
            if chromosome in fitting_models:
                fitting_models[chromosome].append(candidate)
            else:
                fitting_models[chromosome] = [candidate]

    if not fitting_models:
        raise ValueError("no trained models fitting the given parameters were found! Consider training models first (train_ssr_models.py)")
    for chromosome, models in fitting_models.items():
        # models per chromosome only differ in the time stamp. So if sorted, the last model will be the most recently trained
        models.sort()
        fitting_models[chromosome] = os.path.join(path_to_models, models[-1])
    return fitting_models


def predict_self(extragenic, intragenic, val_chromosome, output_name, model_case, extracted_genes):

    x, y, gene_ids = extracted_genes[str(val_chromosome)]

    # Masking
    x[:, extragenic:extragenic + 3, :] = 0                                                                                                  #type:ignore
    x[:, extragenic + (intragenic * 2) + 17:extragenic + (intragenic * 2) + 20, :] = 0                                                      #type:ignore

    newest_model_paths = find_newest_model_path(output_name=output_name, val_chromosome=val_chromosome, model_case=model_case)
    model = load_model(newest_model_paths[val_chromosome])
    pred_probs = model.predict(x).ravel()
    return x, y, pred_probs, gene_ids, model

def predict_other(extragenic, intragenic, val_chromosome, output_name, model_case, extracted_genes) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    x, y, gene_ids = extracted_genes[str(val_chromosome)]

    # Masking
    x[:, extragenic:extragenic + 3, :] = 0                                                                                                  #type:ignore
    x[:, extragenic + (intragenic * 2) + 17:extragenic + (intragenic * 2) + 20, :] = 0                                                      #type:ignore

    newest_model_paths = find_newest_model_path(output_name=output_name, model_case=model_case)
    models = {os.path.basename(model_path): load_model(model_path) for chromosome, model_path in newest_model_paths.items()}


    df_dict = {model_name: model.predict(x).ravel() for model_name, model in models.items()}
    df_dict['true_targets'] = y
    df_dict['genes'] = gene_ids
    result_df = pd.DataFrame(df_dict)
    return result_df, models


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
        extragenic = 1000
        intragenic = 500
        ignore_small_genes = args.ignore_small_genes
        extracted_genes = extract_genes(genome=genome, annotation=annotation, extragenic=extragenic, intragenic=intragenic, ignore_small_genes=ignore_small_genes, tpms=tpms, target_chromosomes=())
        for chrom in range(1, num_chromosomes + 1):
            _, y, pred_probs, gene_ids, _ = predict_self(extragenic=extragenic, intragenic=intragenic, val_chromosome=str(chrom), output_name=output_name,
                                                    model_case=args.model_case, extracted_genes=extracted_genes)
            true_targets.extend(y)
            preds.extend(pred_probs)
            genes.extend(gene_ids)

        result = pd.DataFrame({'true_targets': true_targets, 'pred_probs': preds, 'genes': genes})
        print(result.head())
        output_location = os.path.join(folder_name, f'{output_name}_{file_name}_{get_time_stamp()}.csv')
        result.to_csv(output_location, index=False)

if __name__ == "__main__":
    main()