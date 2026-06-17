import os
from openai import OpenAI
import json

class DossierGenerator:
    """
    Autonomous LLM orchestration agent that translates MOSAIC physics tensor 
    outputs into FDA-compliant pre-clinical scientific dossiers.
    """
    def __init__(self, base_url: str = "https://openrouter.ai/api/v1"):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set. Cannot run LLM Orchestration.")
            
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            default_headers={"HTTP-Referer": "https://mosaic-engine.local", "X-Title": "MOSAIC Pharmacologist"}
        )

    def generate_clinical_dossier(
        self, 
        target_tfs: list[int], 
        dosages: list[float], 
        safety_score: float, 
        efficacy_rate: str, 
        spatial_variance: str
    ):
        """
        Constructs a highly technical prompt injecting the tensor metrics 
        to stream the LLM response.
        """
        
        system_prompt = (
            "You are a Lead AI Pharmacologist and Systems Biologist operating the MOSAIC "
            "thermodynamic cell fate engine. Your job is to translate pure mathematical "
            "tensor outputs from an RBM and Langevin Dynamics simulation into a highly "
            "technical, FDA-compliant Pre-Clinical Dossier.\n\n"
            "Constraints:\n"
            "1. Output MUST be strictly in Markdown format.\n"
            "2. DO NOT hallucinate biological mechanisms that contradict the provided numerical metrics.\n"
            "3. Ground all biological explanations in the context of Thermodynamic Free Energy, "
            "Attractor Basins, and spatial paracrine interactions."
        )
        
        user_prompt = (
            "Please generate the 'Pre-Clinical Pharmacodynamics & Safety Dossier' for the "
            "following autonomously discovered multi-gene intervention:\n\n"
            "**Intervention Metrics:**\n"
            f"- Target Transcription Factor Indices: {target_tfs}\n"
            f"- Calculated Continuous Dosages (Δv): {dosages}\n"
            f"- Pleiotropy Safety Score (0-100): {safety_score:.1f}\n"
            f"- Virtual Cohort Efficacy Rate: {efficacy_rate}\n"
            f"- Spatial Paracrine $\Delta$E Variance: {spatial_variance}\n\n"
            "The dossier must include exactly the following sections:\n"
            "## 1. Executive Summary\n"
            "## 2. Thermodynamic Mechanism of Action\n"
            "*(Interpret the RBM energy shift causing the cell to move from origin to target basin)*\n"
            "## 3. Spatial Paracrine Safety Profile\n"
            "*(Interpret the shockwave / collateral pleiotropy and safety score)*\n"
            "## 4. Virtual Cohort Efficacy Analysis\n"
            "*(Interpret the robustness across the 1,000 virtual patient cohort)*\n"
        )
        
        response_stream = self.client.chat.completions.create(
            model="anthropic/claude-3-haiku", # Highly technical and fast model via OpenRouter
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            stream=True,
            temperature=0.2
        )
        
        return response_stream
