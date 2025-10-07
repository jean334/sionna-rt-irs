# SPDX-FileCopyrightText: Copyright (c) 2025 Jean ACKER.
# SPDX-License-Identifier: Apache-2.0

import os
# CUDA and GPU settings
os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir=/usr/local/cuda-12.5"

if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0 # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from numpy import pi as PI

from irs.ris import RIS
from sionna.rt import Camera
from rt import PlanarArray
from rt import Transmitter, Receiver, RadioMapSolver
from rt import load_scene


"""
Set the scene
"""
scene = load_scene("./scenes/simple_scene.xml")

scene.frequency = 3.5e9 # Carrier frequency [Hz]
scene.tx_array = PlanarArray(num_rows=8,
                             num_cols=4,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="tr38901",
                             polarization="VH")


scene.rx_array = PlanarArray(num_rows=1,
                            num_cols=1,
                            vertical_spacing=0.5,
                            horizontal_spacing=0.5,
                            pattern="iso",
                            polarization="V")


"""
add tx, rx, ris
"""
scene.remove("tx")
scene.remove("rx")
scene.remove("ris")

tx = Transmitter("tx", position=[-8,1,0], look_at=[0,2,0])
scene.add(tx)

rx = Receiver("rx", position=[-4,1,0])
ris = RIS(name="ris",position=[0,2,0],orientation=[0,-PI,0],num_rows=50,num_cols=50,frequency=scene.frequency, num_modes=1, display_radius=.5, color=(1,0,1))
ris.frequency = scene.frequency

scene.add(ris)

#sources = [[-8, -8],[1, 1],[-2, -2]]
#targets = [[-4, -4],[1, 1],[12, 12]]
sources = [[-8],[1],[0]]
targets = [[-4],[1],[6]]

"""
init the phase profile of the RIS to reflect from sources to targets
"""
ris.phase_gradient_reflector(sources, targets)
#ris.uniform_init() #with this profile, the IRS act as a mirror

"""
Phase and amplitude profile visualization (see ./results)
"""
ris.amplitude_profile.show()
ris.phase_profile.show()

"""
Camera setting
"""
bird_cam = Camera(position=[0,22,0], orientation=np.array([-PI/2,0,-PI/2]))
cam2 = Camera(position=[-3,6,-15], orientation=np.array([-PI/2,-PI/2,0]))
save_dir = "results"

scene.render_to_file(camera=bird_cam,
                        fov=90,
                        filename=os.path.join(save_dir, f"scene.png")
                        )

scene.render_to_file(camera=cam2,
                        fov=90,
                        filename=os.path.join(save_dir, f"scene_cam2.png")
                        )

"""
Radio map setting and computation
"""
cm_center = [0, 1, 0]            # Center of the coverage map
cm_orientation = [PI/2, 0, 0]    # Orientation of the coverage map
cm_size = [20, 20]               # Size of the coverage map
cm_cell_size = [.2, .2]          # Cell size of the coverage map
cm_max_depth = 2                 # Max depth of the rays
cm_samples_per_tx = 1000000      # Number of rays launched per transmitter

solver = RadioMapSolver()

rm = solver(scene, 
            center= cm_center,
            orientation= cm_orientation,
            size= cm_size,
            cell_size= cm_cell_size, 
            max_depth= cm_max_depth,
            samples_per_tx= cm_samples_per_tx)

"""
Render radio map from two camera views (see ./results)
"""
scene.render_to_file(camera=bird_cam,
                        fov=90,
                        radio_map=rm,
                        rm_metric="rss",
                        rm_db_scale = True,
                        rm_vmax=0,
                        rm_vmin=-60,
                        filename=os.path.join(save_dir, f"scene_rm_bird_cam.png"))

scene.render_to_file(camera=cam2,
                        fov=90,
                        radio_map=rm,
                        rm_metric="rss",
                        rm_db_scale = True,
                        rm_vmax=0,
                        rm_vmin=-60,
                        filename=os.path.join(save_dir, f"scene_rm_cam2.png"))


