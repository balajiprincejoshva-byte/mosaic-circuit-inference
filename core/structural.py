import requests

class AlphaFoldBridge:
    """
    Autonomously fetches 3D protein conformation (.pdb) data for 
    optimized transcription factors from the AlphaFold DB.
    """
    
    # Standard mapping for key TFs (UniProt IDs)
    UNIPROT_MAP = {
        'MYC': 'P01106',
        'SOX2': 'P48431',
        'POU5F1': 'Q01860',
        'KLF4': 'O43474',
        'TP53': 'P04637',
        'STAT3': 'P40763',
        'GATA3': 'P23771',
        'FOXA1': 'P55317',
        'ESR1': 'P03372',
        'HNF4A': 'P41235'
    }

    def fetch_protein_structure(self, gene_name: str) -> str:
        """
        Fetches the .pdb string from AlphaFold DB for a given gene.
        """
        gene_name = gene_name.upper()
        uniprot_id = self.UNIPROT_MAP.get(gene_name)
        
        if not uniprot_id:
            raise ValueError(f"Gene '{gene_name}' not found in AlphaFoldBridge mapping. Available: {list(self.UNIPROT_MAP.keys())}")
            
        # Try fetching the dynamically versioned pdbUrl from the EBI API first
        api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
        url = None
        try:
            api_response = requests.get(api_url, timeout=5)
            if api_response.status_code == 200:
                predictions = api_response.json()
                if predictions and isinstance(predictions, list) and len(predictions) > 0:
                    url = predictions[0].get("pdbUrl")
        except Exception:
            pass
            
        # Fallback to the classic v4 endpoint if API query fails or pdbUrl is not found
        if not url:
            url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to fetch AlphaFold structure for {gene_name} ({uniprot_id}) from {url}: {e}")

