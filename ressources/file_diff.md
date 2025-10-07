### `rt`
**Diff**
- `renderer.py`
	- + `from .utils.irs_utils import mitsuba_rectangle_to_world`
	- + `for prefix, irs, enabled in (('irs', scene.irs, show_targets),):` --> representing IRS as a mitsuba rectangle
- `scene.py`
	- + `def wavenumber(self):` --> adding getter for wavenumber
	- Adding RIS support to the `Scene` class
		- + l. 252 `def irs(self):`
		- + l. 289 
		- + l. 326
		- + l. 952
---
### `rt/utils`
**Add**
- `irs_misc.py`
- `irs_tensors.py`
- `irs_utils.py`
- `legacy_utils.py`
---
### `rt/radio_map_solvers`
**Diff**
- `radio_map.py`
	- + l. 21 --> import
	- + l. 86 --> init RIS in `RadioMap.__init__`
	- + l. 212 --> number of RIS getter
	- + l. 243 --> Computes and returns the cell index positions corresponding to IRS
	- + l. 336 --> modified `add` method
	- + l. 440 --> modified `show` method
- `radio_map_solver.py`
	- + l.11 --> import
	- + l.203 --> `DISCARD_THRES`
	- + l. 232 --> `_build_mi_ris_objects` method : Builds a Mitsuba scene containing all RIS as rectangles with position, orientation, and size matching the RIS properties
	- + l. 266 --> rewriting `RadioMapSolver.__call__` to take IRS into account
	- + l. 451 --> rewriting `shoot_and_bounce` method
	- + l. 732 --> writing `_ris_intersect` method
	- + l. 762 --> writing `_compute_ris_reflected_field` method

### `rt/radio_materials`
**Change**
- `radio_materials_base.py`
	- + l. 13 --> import

### `rt/path_solver`
**Unchanged**

---
### `rt/radio_devices`
**Unchanged**

---
### `rt/radio_materials`
**Unchanged**

---
### `rt/scenes`
**Unchanged**

---
### `irs`
**Add**
- `ris.py`
- `config_legacy.py`
- `solver_paths_legacy.py`
- `paths_legacy.py`


