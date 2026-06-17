import requests

class AlphaFoldBridge:
    """
    Autonomously fetches 3D protein conformation (.pdb) data for 
    optimized transcription factors from the AlphaFold DB.
    """
    
    def resolve_uniprot_id(self, gene_name: str) -> str:
        """
        Dynamically queries the public UniProt REST API to resolve a human gene symbol
        to its primary UniProt accession ID.
        """
        url = f"https://rest.uniprot.org/uniprotkb/search?query=gene_exact:{gene_name}+AND+organism_id:9606"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('results') and len(data['results']) > 0:
                # Return the primary accession of the top hit
                return data['results'][0]['primaryAccession']
                
        raise ValueError(f"Gene '{gene_name}' could not be resolved to a UniProt ID via REST API.")

    def fetch_protein_structure(self, gene_name: str) -> str:
        """
        Fetches the .pdb string from AlphaFold DB for a given gene.
        """
        gene_name = gene_name.upper()
        uniprot_id = self.resolve_uniprot_id(gene_name)
            
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

