## IRS Implementation in Sionna RT

---

This repository is based on **Sionna RT**, the ray-tracing package from [Sionna](https://nvlabs.github.io/sionna/), version **1.0.2**.
This repository includes original components from NVIDIA Sionna RT (Apache 2.0 License)
and new code developed by Jean ACKER under the same license.
---

### Installation

---

1. First, install **Sionna RT** or **Sionna**:
    
    ```bash
    pip install sionna-rt
    # OR
    pip install sionna
    ```
    
2. An installation guide is available directly on the [Sionna RT GitHub repository](https://github.com/NVlabs/sionna-rt/tree/main).
    

This repository introduces two main folders:
- `irs/`
- `rt/`

The `irs/` folder is **new** (not present in Sionna RT v1.0.2).  
The `rt/` folder is mostly a copy of Sionna RT’s original folder, but several files have been modified.  
These modified files are highlighted in the diagram below, and a detailed changelog is available in [[file_diff.md]].

```
├── irs/
│   ├── ris.py
│   ├── config_legacy.py
│   ├── solver_paths_legacy.py
│   └── paths_legacy.py
└── rt/
    ├── renderer.py
    ├── scene.py
    ├── utils/
    │   ├── irs_misc.py
    │   ├── irs_tensors.py
    │   ├── irs_utils.py
    │   └── legacy_utils.py
    ├── radio_map_solvers/
    │   ├── radio_map.py
    │   └── radio_map_solver.py
    └── radio_materials/
        └── radio_materials_base.py
```

The simplest way to use the IRS in your Sionna scene is to **copy the `irs/` and `rt/` folders** into your working directory, then update your imports accordingly.  
For example:

```python
from sionna.rt import PlanarArray
# becomes
from rt import PlanarArray
```
An example usage is provided in [[phase_gradient_test.py]].

---
### How the IRS Is Integrated into Sionna RT

---
- IRS functionality existed in older versions of `sionna-rt` (up to `v0.19`) but was removed in `v1.0.0`, when Sionna transitioned from **TensorFlow** to **Dr.Jit** for tensor computations.
    
- The **IRS class** and all related components have been **completely rewritten** to use **Dr.Jit** instead of TensorFlow.
    
- The `Scene` class has been extended to support IRS objects.
    
- The `Renderer` has been updated to visualize IRS as planar elements.
    
- Most of the ray-tracing logic resides in the `RadioMapSolver`.  
    Since IRS objects are _responsive components_, they cannot easily be modeled as standard BSDFs.  
    To integrate them into the scene, **Mitsuba rectangles** are instantiated at the IRS positions and orientations.  
    **Masks** are then used to determine which rays hit which part of the scene first (e.g., a wall or an IRS).
    
- Rays that hit the scene or the IRS first are treated separately, even though they are stored in the same tensor structures.
    
- New rays are spawned from the intersection points based on the computed reflection or scattering directions.
    
- This cycle repeats until the specified maximum recursion depth or threshold is reached.
    

A **diagram summarizing this algorithm** and the main method calls is provided in `ressources/code_diagram.png` and `ressources/RadioMapSolver.canvas`.

### Example
--- 

A complete example is provided with this repository:
`phase_gradient_test.py` (and `phase_gradient_test_gif.py`, used to generate the GIFs in `resources/phase_profile.gif`, `resources/rm_bird_cam.gif`, and `resources/rm_cam2.gif`).

In this example, an IRS with a 2 × 2 m surface is placed 2 m above the ground (represented as a purple plane in the GIF).
The IRS itself remains fixed, but its phase profile changes dynamically, steering the reflected waves to sweep across the scene from left to right.

The red dot represents the transmitter (Tx), located 1 m above the ground.
The radio map is also measured 1 m above the ground, providing a visualization of the received power distribution.

A GIF animation is available, showing the temporal evolution of the IRS phase profile and its impact on the propagation in the scene.


### Citation
---
If you use this code in your research, please cite both **Sionna RT** and this work.

#### Sionna RT

Sionna RT is part of the Sionna library developed by NVIDIA.  
If you use the ray-tracing features of this repository, please cite:

```
@software{sionna,
 title = {Sionna},
 author = {Hoydis, Jakob and Cammerer, Sebastian and {Ait Aoudia}, Fayçal and Nimier-David, Merlin and Maggi, Lorenzo and Marcus, Guillermo and Vem, Avinash and Keller, Alexander},
 note = {https://nvlabs.github.io/sionna/},
 year = {2022},
 version = {1.2.0}
}
```

#### IRS Extension

If you use or reference the IRS implementation provided in this repository, please cite:

```
@software{jeanACKER2025irs,   
 author    = {Jean ACKER},   
 title     = {IRS Extension for Sionna RT},   
 year      = {2025},   
 url       = {https://github.com/cir-lab/irs-sionna-rt},   
 note      = {An extension of Sionna RT enabling Intelligent Reflecting Surface (IRS) simulation and optimization.} 
 }
```
