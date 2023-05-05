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
model_2_path = input_path / 'model_2.csv'

travel_curve_model_1 = output_path / 'a010_travel_curves.png'
slowness_surface_1 = output_path / 'a010_slowness_surface.png'


travel_curve_model_2 = output_path / 'a020_travel_curves.png'
slowness_surface_2 = output_path / 'a020_slowness_surface.png'