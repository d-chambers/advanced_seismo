"""
Project for reproducing tromp et al., 2005 figures
"""
from pathlib import Path

import obspy

# setup paths
here = Path(__file__).absolute().parent
input_path = here / Path('inputs')
output_path = here / Path('outputs')
output_path.mkdir(exist_ok=True)

model_1_path = input_path / 'model_1.csv'
