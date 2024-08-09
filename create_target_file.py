import pandas as pd
import numpy as np

tpm_counts = pd.read_csv('tpm_counts/arabidopsis_counts.csv')
true_targets = []

for log_count in tpm_counts['logMaxTPM'].values:
    if log_count <= np.percentile(tpm_counts['logMaxTPM'], 25):
        true_targets.append(0)
    elif log_count >= np.percentile(tpm_counts['logMaxTPM'], 75):
        true_targets.append(1)
    else:
        true_targets.append(2)
tpm_counts['target'] = true_targets
tpm_counts = tpm_counts[['gene_id', 'target']]
tpm_counts.to_csv(path_or_buf='tpm_counts/arabidopsis_targets.csv', index=False)
print(tpm_counts.head())
