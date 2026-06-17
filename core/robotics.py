import textwrap

class OpentronsCompiler:
    """
    Compiles optimal target vectors into executable OpenTrons (OT-2) Python protocols.
    Bridges the digital physics engine to wet-lab liquid handling robots.
    """
    def __init__(self):
        # Define standard labware for physical translation
        self.plate_type = "corning_96_wellplate_360ul_flat"
        self.tiprack_type = "opentrons_96_tiprack_300ul"
        self.pipette_type = "p300_single_gen2"
        self.reagent_rack_type = "opentrons_24_tuberack_generic_2ml_screwcap"
        
    def generate_dispense_protocol(self, target_tfs: list, dosages: list) -> str:
        """
        Generates a string representing a valid OT-2 Python protocol.
        Translates dimensionless dosages into physical microliter (uL) dispense volumes.
        """
        transfers = []
        
        # Scaling factor: maximum dosage maps to 50 uL dispense
        max_dosage = max(dosages) if dosages else 1.0
        if max_dosage == 0: max_dosage = 1.0
        
        for i, (tf_id, dosage) in enumerate(zip(target_tfs, dosages)):
            volume = (dosage / max_dosage) * 50.0
            volume = max(1.0, round(volume, 1)) # Minimum 1.0 uL for realistic OT-2 handling
            
            # Map index to 96-well format (A1 to H12)
            row = chr(65 + (i % 8)) # A-H
            col = (i // 8) + 1      # 1-12
            dest_well = f"{row}{col}"
            
            # Reagent source well (mocked as distinct wells in the 24-tube rack)
            src_row = chr(65 + (i % 4))
            src_col = (i // 4) + 1
            src_well = f"{src_row}{src_col}"
            
            transfers.append(
                f"    # Target {i+1}: Transfer {volume}uL of Compound {tf_id} to {dest_well}\n"
                f"    pipette.transfer({volume}, reagent_rack['{src_well}'], plate['{dest_well}'], new_tip='always')"
            )
            
        transfers_str = "\n".join(transfers)
        
        protocol_template = f"""from opentrons import protocol_api

metadata = {{
    'apiLevel': '2.13',
    'protocolName': 'MOSAIC Automated Tissue Perturbation',
    'description': 'Autonomously generated protocol for precision biological perturbation.',
    'author': 'MOSAIC Platform'
}}

def run(protocol: protocol_api.ProtocolContext):
    # Load Labware
    plate = protocol.load_labware('{self.plate_type}', 1)
    tiprack = protocol.load_labware('{self.tiprack_type}', 2)
    reagent_rack = protocol.load_labware('{self.reagent_rack_type}', 3)
    
    # Load Instruments
    pipette = protocol.load_instrument('{self.pipette_type}', 'right', tip_racks=[tiprack])
    
    # Execute Pharmacological Perturbations
{transfers_str}
    
    protocol.comment("Perturbation dispense complete. Proceed to incubation.")
"""
        return protocol_template
