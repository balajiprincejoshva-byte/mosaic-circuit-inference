import numpy as np
from typing import Dict, List, Tuple

class FactorGraphBP:
    """
    Probabilistic Graphical Model (PGM) for inferring causal regulatory topology.
    Implements Loopy Belief Propagation via matrix operations.
    Nodes: TF (protein), Enhancer (atac), Gene (rna)
    Directed arc: TF -> Enhancer -> Gene
    """
    def __init__(self, num_tfs: int, num_enhancers: int, num_genes: int):
        self.num_tfs = num_tfs
        self.num_enhancers = num_enhancers
        self.num_genes = num_genes
        
        # Dimensions for messages:
        # TF states: active/inactive (2)
        # Enhancer states: open/closed (2)
        # Gene states: expressed/unexpressed (2)
        self.num_states = 2
        
        # Prior probabilities (Unary potentials) initialized uniformly
        # Shape: (Num_Nodes, Num_States)
        self.phi_tf = np.ones((num_tfs, self.num_states)) / self.num_states
        self.phi_enh = np.ones((num_enhancers, self.num_states)) / self.num_states
        self.phi_gene = np.ones((num_genes, self.num_states)) / self.num_states

        # Pairwise potentials (Edge potentials)
        # For simplicity, assuming a common prior potential for interactions that favors
        # (active -> open) and (open -> expressed)
        # Shape: (Num_States_Source, Num_States_Target)
        self.psi_tf_enh = np.array([[0.8, 0.2], [0.2, 0.8]]) # [inactive->closed, inactive->open], [active->closed, active->open]
        self.psi_enh_gene = np.array([[0.8, 0.2], [0.2, 0.8]])
        
        # Candidate Adjacency matrices (binary or probabilities). 
        # Initialized completely connected for message passing (could be sparsified based on biological priors)
        self.adj_tf_enh = np.ones((num_tfs, num_enhancers))
        self.adj_enh_gene = np.ones((num_enhancers, num_genes))
        
        # Initialize messages (log space for numerical stability, or probability space. Let's use probability space for now)
        # Message from TF to Enhancer
        self.msg_tf_enh = np.ones((num_tfs, num_enhancers, self.num_states)) / self.num_states
        # Message from Enhancer to Gene
        self.msg_enh_gene = np.ones((num_enhancers, num_genes, self.num_states)) / self.num_states
        
        # Backward messages
        self.msg_enh_tf = np.ones((num_enhancers, num_tfs, self.num_states)) / self.num_states
        self.msg_gene_enh = np.ones((num_genes, num_enhancers, self.num_states)) / self.num_states

    def set_priors(self, tf_priors: np.ndarray, enh_priors: np.ndarray, gene_priors: np.ndarray):
        """
        Set unary potentials based on observed data.
        Priors should be shape (N, 2) where col 0 is prob inactive, col 1 is prob active.
        """
        assert tf_priors.shape == (self.num_tfs, 2)
        assert enh_priors.shape == (self.num_enhancers, 2)
        assert gene_priors.shape == (self.num_genes, 2)
        
        self.phi_tf = np.clip(tf_priors, 1e-9, 1.0)
        self.phi_enh = np.clip(enh_priors, 1e-9, 1.0)
        self.phi_gene = np.clip(gene_priors, 1e-9, 1.0)
        
        self._normalize(self.phi_tf, axis=1)
        self._normalize(self.phi_enh, axis=1)
        self._normalize(self.phi_gene, axis=1)

    def _normalize(self, matrix: np.ndarray, axis: int = -1):
        """In-place normalization of probability distributions."""
        sums = matrix.sum(axis=axis, keepdims=True)
        sums[sums == 0] = 1.0
        matrix /= sums

    def run_loopy_bp(self, max_iters: int = 50, tol: float = 1e-4):
        """
        Execute Loopy Belief Propagation using matrix multiplication.
        """
        for iteration in range(max_iters):
            # Store old messages for convergence check
            old_msg_tf_enh = self.msg_tf_enh.copy()
            
            # 1. Update Messages TF -> Enhancer
            # TF sends its prior * product of incoming messages from OTHER enhancers (if any feedback).
            # For a pure directed model without backward cycles, msg_tf_enh is just based on prior and psi.
            # However, this is a loopy PGM. We include msg_enh_tf
            
            # belief_tf_excluding_j = phi_tf(x) * prod_{k != j} msg_enh_k_to_tf_i
            # To compute prod_{k != j}, we do prod_all / msg_j. In log space it's sum_all - msg_j.
            
            # Log space for stability
            log_phi_tf = np.log(self.phi_tf) # (T, 2)
            log_msg_enh_tf = np.log(np.clip(self.msg_enh_tf, 1e-9, 1.0)) # (E, T, 2)
            
            # Sum incoming to TF: (T, 2)
            sum_in_tf = log_msg_enh_tf.sum(axis=0)
            
            # Compute outgoing TF -> Enhancer (T, E, 2)
            # sum_in_tf (T, 2) - log_msg_enh_tf.transpose(1, 0, 2) (T, E, 2) + log_phi_tf (T, 2) -> (T, E, 2)
            out_tf_log = sum_in_tf[:, None, :] - log_msg_enh_tf.transpose(1, 0, 2) + log_phi_tf[:, None, :]
            out_tf_prob = np.exp(out_tf_log)
            
            # Convolve with edge potential psi_tf_enh
            # m_{i->j}(x_j) = sum_{x_i} psi(x_i, x_j) * out_tf_prob(x_i)
            # psi is (2, 2). out_tf_prob is (T, E, 2)
            # Matrix mult: (T, E, 2) @ (2, 2) -> (T, E, 2)
            self.msg_tf_enh = out_tf_prob @ self.psi_tf_enh
            self._normalize(self.msg_tf_enh)
            
            # 2. Update Messages Enhancer -> Gene
            log_phi_enh = np.log(self.phi_enh) # (E, 2)
            log_msg_tf_enh = np.log(np.clip(self.msg_tf_enh, 1e-9, 1.0)) # (T, E, 2)
            log_msg_gene_enh = np.log(np.clip(self.msg_gene_enh, 1e-9, 1.0)) # (G, E, 2)
            
            sum_in_enh = log_msg_tf_enh.sum(axis=0) + log_msg_gene_enh.sum(axis=0) # (E, 2)
            
            out_enh_log = sum_in_enh[:, None, :] - log_msg_gene_enh.transpose(1, 0, 2) + log_phi_enh[:, None, :] # (E, G, 2)
            out_enh_prob = np.exp(out_enh_log)
            
            self.msg_enh_gene = out_enh_prob @ self.psi_enh_gene
            self._normalize(self.msg_enh_gene)
            
            # 3. Update Messages Gene -> Enhancer
            log_phi_gene = np.log(self.phi_gene) # (G, 2)
            sum_in_gene = np.log(np.clip(self.msg_enh_gene, 1e-9, 1.0)).sum(axis=0) # (G, 2)
            
            out_gene_log = sum_in_gene[:, None, :] - np.log(np.clip(self.msg_enh_gene.transpose(1, 0, 2), 1e-9, 1.0)) + log_phi_gene[:, None, :] # (G, E, 2)
            out_gene_prob = np.exp(out_gene_log)
            
            # Transpose psi for backward message
            self.msg_gene_enh = out_gene_prob @ self.psi_enh_gene.T
            self._normalize(self.msg_gene_enh)
            
            # 4. Update Messages Enhancer -> TF
            out_enh_backward_log = sum_in_enh[:, None, :] - log_msg_tf_enh.transpose(1, 0, 2) + log_phi_enh[:, None, :] # (E, T, 2)
            out_enh_backward_prob = np.exp(out_enh_backward_log)
            
            self.msg_enh_tf = out_enh_backward_prob @ self.psi_tf_enh.T
            self._normalize(self.msg_enh_tf)
            
            # Check convergence
            delta = np.max(np.abs(self.msg_tf_enh - old_msg_tf_enh))
            if delta < tol:
                break

    def extract_circuits(self, threshold: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate posterior probabilities of edges (beliefs) and return adjacency matrices.
        Returns:
            adj_tf_enh: (T, E) binary matrix
            adj_enh_gene: (E, G) binary matrix
        """
        # Edge Belief b(x_i, x_j) proportional to phi(x_i)*phi(x_j)*psi(x_i,x_j)*prod(msgs_to_i)*prod(msgs_to_j)
        
        # TF-Enhancer edge belief
        # To simplify, we calculate the expected value of the active-active edge state.
        # Active-active is index (1, 1) in the joint 2x2 distribution.
        
        # Calculate node marginals (beliefs) first
        log_phi_tf = np.log(self.phi_tf)
        log_msg_enh_tf = np.log(np.clip(self.msg_enh_tf, 1e-9, 1.0))
        belief_tf = np.exp(log_phi_tf + log_msg_enh_tf.sum(axis=0))
        self._normalize(belief_tf, axis=1)
        
        log_phi_enh = np.log(self.phi_enh)
        log_msg_tf_enh = np.log(np.clip(self.msg_tf_enh, 1e-9, 1.0))
        log_msg_gene_enh = np.log(np.clip(self.msg_gene_enh, 1e-9, 1.0))
        belief_enh = np.exp(log_phi_enh + log_msg_tf_enh.sum(axis=0) + log_msg_gene_enh.sum(axis=0))
        self._normalize(belief_enh, axis=1)
        
        log_phi_gene = np.log(self.phi_gene)
        belief_gene = np.exp(log_phi_gene + np.log(np.clip(self.msg_enh_gene, 1e-9, 1.0)).sum(axis=0))
        self._normalize(belief_gene, axis=1)
        
        # Calculate simplified edge connectivity likelihood based on joint active probabilities
        # A naive but fast way is to just multiply the marginal probability of being active 
        # (index 1) for both source and target.
        # Since we initialized the graph fully connected, this reveals which edges are strongly supported.
        
        prob_tf_active = belief_tf[:, 1]
        prob_enh_active = belief_enh[:, 1]
        prob_gene_active = belief_gene[:, 1]
        
        # Heuristic edge score: joint probability assuming independence given the local beliefs
        score_tf_enh = np.outer(prob_tf_active, prob_enh_active) * self.psi_tf_enh[1, 1]
        score_enh_gene = np.outer(prob_enh_active, prob_gene_active) * self.psi_enh_gene[1, 1]
        
        # Threshold to create sparse adjacency matrices
        adj_tf_enh = (score_tf_enh > threshold).astype(int)
        adj_enh_gene = (score_enh_gene > threshold).astype(int)
        
        return adj_tf_enh, adj_enh_gene
