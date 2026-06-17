import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
from typing import Dict, Union, Tuple, List

class MultiOmicTensor:
    """
    High-performance custom data loader for single-cell multi-omics.
    Wraps scipy.sparse.csr_matrix to prevent memory explosion.
    """
    def __init__(self, data_dict: Dict[str, Union[sp.csr_matrix, np.ndarray, torch.Tensor]]):
        """
        Initialize with a dictionary of omics modalities.
        Example: {'rna': rna_matrix, 'atac': atac_matrix, 'adt': adt_matrix}
        """
        self.data = {}
        for modality, mat in data_dict.items():
            if isinstance(mat, sp.csr_matrix):
                self.data[modality] = mat
            elif isinstance(mat, np.ndarray):
                self.data[modality] = sp.csr_matrix(mat)
            elif isinstance(mat, torch.Tensor):
                self.data[modality] = sp.csr_matrix(mat.numpy())
            else:
                raise ValueError(f"Unsupported matrix type for {modality}: {type(mat)}")

    def normalize_rna(self) -> None:
        """
        Applies log1p(CPM) normalization to RNA modality.
        CPM: Counts Per Million.
        """
        if 'rna' not in self.data:
            return
        
        rna_mat = self.data['rna']
        # Compute row sums (total counts per cell)
        cell_totals = np.array(rna_mat.sum(axis=1)).squeeze()
        
        # Avoid division by zero
        cell_totals[cell_totals == 0] = 1.0
        
        # Scale to CPM (1e6)
        scaling_factors = 1e6 / cell_totals
        
        # Sparse element-wise multiplication by scaling factors per row
        # We can use scipy.sparse diags
        diag_scaler = sp.diags(scaling_factors)
        cpm_mat = diag_scaler.dot(rna_mat)
        
        # Apply log1p: since sparse matrix log1p is supported in scipy > 1.2.0 for csr
        self.data['rna'] = cpm_mat.log1p()

    def normalize_atac(self) -> None:
        """
        Applies TF-IDF (Term Frequency-Inverse Document Frequency) normalization to ATAC peaks.
        """
        if 'atac' not in self.data:
            return
            
        atac_mat = self.data['atac']
        
        # Term Frequency (TF): peak counts divided by total peaks in cell
        cell_totals = np.array(atac_mat.sum(axis=1)).squeeze()
        cell_totals[cell_totals == 0] = 1.0
        tf_mat = sp.diags(1.0 / cell_totals).dot(atac_mat)
        
        # Inverse Document Frequency (IDF): log(total_cells / cells_with_peak)
        num_cells = atac_mat.shape[0]
        # Count non-zero entries per column
        cells_with_peak = np.array((atac_mat > 0).sum(axis=0)).squeeze()
        cells_with_peak[cells_with_peak == 0] = 1.0
        
        idf = np.log(num_cells / cells_with_peak)
        
        # Multiply TF by IDF
        # tf_mat is CSR, we multiply columns by idf
        tfidf_mat = tf_mat.dot(sp.diags(idf))
        
        self.data['atac'] = tfidf_mat

    def normalize_adt(self) -> None:
        """
        Applies CLR (Centered Log-Ratio) normalization to CITE-seq ADT (protein) modality.
        CLR(x) = log(x / geometric_mean(x))
        Computed across cells (margin=0) or across proteins (margin=1). We will do across proteins.
        """
        if 'adt' not in self.data:
            return
            
        adt_mat = self.data['adt']
        
        # Add a pseudo-count of 1 to avoid log(0)
        # CLR usually requires dense conversion if we add 1 everywhere, 
        # but proteins are usually low-dimensional so dense is fine.
        dense_adt = adt_mat.toarray() + 1.0
        
        # Compute geometric mean across proteins (axis=1)
        # geomean = exp(mean(log(x)))
        log_adt = np.log(dense_adt)
        mean_log = np.mean(log_adt, axis=1, keepdims=True)
        
        clr_mat = log_adt - mean_log
        
        # Convert back to sparse to maintain interface, even if dense
        self.data['adt'] = sp.csr_matrix(clr_mat)

    def to_tensor(self, modalities: List[str] = ['rna', 'atac', 'adt']) -> torch.Tensor:
        """
        Concatenates requested modalities and returns a dense PyTorch tensor.
        Useful for feeding into the RBM.
        """
        tensors = []
        for mod in modalities:
            if mod in self.data:
                dense_arr = self.data[mod].toarray()
                tensors.append(torch.tensor(dense_arr, dtype=torch.float32))
        
        if not tensors:
            raise ValueError("No matching modalities found to convert to tensor.")
            
        return torch.cat(tensors, dim=1)

    @property
    def shape(self) -> Tuple[int, int]:
        if not self.data:
            return (0, 0)
        
        num_cells = list(self.data.values())[0].shape[0]
        num_features = sum(mat.shape[1] for mat in self.data.values())
        return (num_cells, num_features)
