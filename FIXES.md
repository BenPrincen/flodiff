# Fixes Applied

This document records fixes applied to get the flodiff evaluation script and iGibson build working.

---

## 1. iGibson build (`pip install -e .` in iGibson)

### 1.1 Missing Git submodules
**Problem:** CMake failed with “The source directory …/pybind11 does not contain a CMakeLists.txt file” (and same for cryptopp, glfw, glm).

**Cause:** Repo was cloned without `--recursive`, so submodules were not checked out.

**Fix:** In the iGibson repo root:
```bash
cd /path/to/iGibson
git submodule update --init --recursive
```

### 1.2 Python 3.13 incompatibility
**Problem:** Build failed with pybind11 errors (`PyFrameObject`, `function_record` members like `nargs`, `std::uint16_t`).

**Cause:** iGibson’s bundled pybind11 is old and does not support Python 3.12+.

**Fix:** Use Python 3.8–3.11. The flodiff conda env (Python 3.8) is used for both flodiff and iGibson.

### 1.3 pybind11 `std::uint16_t` not in scope
**Problem:** With Python 3.8, build still failed: `'uint16_t' in namespace 'std' does not name a type` in pybind11 headers.

**Fix:** In **iGibson** (not flodiff) at  
`igibson/render/pybind11/include/pybind11/attr.h`, add after `#pragma once`:
```cpp
#include <cstdint>
```

---

## 2. Config and paths

### 2.1 test_flona_gtpos.py in iGibson scripts
**File:** `iGibson/igibson/scripts/test_flona_gtpos.py`

**Changes:**
- `sys.path.append('/path/to/flodiff')` → `sys.path.append('/home/vgmachinist/projects/flodiff')`
- `config_path = '/path/to/flodiff/test.yaml'` → `config_path = '/home/vgmachinist/projects/flodiff/test.yaml'`

Adjust paths if your flodiff repo lives elsewhere.

### 2.2 test.yaml scene_path
**Problem:** Script failed with “Scene Azusa does not exist”.  
**Cause:** `scene_path` pointed to `.../flodiff/iGibson/...`, but iGibson (and Gibson data) live at `.../iGibson/...`.

**Fix:** In **flodiff/test.yaml**:
```yaml
scene_path: '/home/vgmachinist/projects/iGibson/igibson/data/gibson_v2_selected/'
```

### 2.3 Gibson dataset path at runtime
**Note:** iGibson loads scenes via its own `g_dataset_path` (from `global_config.yaml` or `GIBSON_DATASET_PATH`), not from `test.yaml`’s `scene_path`. The script uses `scene_path` only for things like `scene_dir`.

**Fix:** When running the script, set:
```bash
export GIBSON_DATASET_PATH=/home/vgmachinist/projects/iGibson/igibson/data/gibson_v2_selected
```
(or the path where your Gibson v2 selected scenes are).

---

## 3. Traversability map filename

**Problem:** `FileNotFoundError: ... floor_trav_test_0_modified_8bit.png`  
**Cause:** Script expected `*_modified_8bit.png`; dataset has `*_modified.png`.

**Fix:** In **iGibson/igibson/scripts/test_flona_gtpos.py**, when building the traversability map path, try the 8bit path first and fall back to the non-8bit path:
- Define `travers_path_8bit` and `travers_path_fallback` (same as current logic but with and without `_8bit`).
- Set `travers_path = travers_path_8bit if os.path.exists(travers_path_8bit) else travers_path_fallback` before loading the image.

---

## 4. save_state and collision (NumPy/boolean)

**Problem:**  
- `ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape` when appending to `save_state`.  
- `ValueError: The truth value of an array with more than one element is ambiguous` on `if collision:`.

**Cause:** `check_collision()` can return a numpy array or 0-d array; using it in `np.array([...])` or in a boolean context caused shape/ambiguity errors.

**Fix:** In **iGibson/igibson/scripts/test_flona_gtpos.py**:

1. **Where collision is assigned** (after each `check_collision(...)` call), normalize to a scalar boolean:
   ```python
   collision = bool(np.asarray(check_collision(...)).any())
   ```

2. **In all `save_state.append(...)`** in that script, use scalar values and collision as a bool:
   ```python
   save_state.append(np.array([
       float(current_position[0]),
       float(current_position[1]),
       float(current_heading_point[0]),
       float(current_heading_point[1]),
       bool(np.asarray(collision).any())
   ]))
   ```
   (Once collision is always set via the line above, the last element can be `collision` instead of `bool(np.asarray(collision).any())`.)

---

## 5. Headless mode and viewer

**Problem:**  
- With a display: Qt error “Could not load the Qt platform plugin 'xcb'” and abort.  
- In headless: `AttributeError: 'NoneType' object has no attribute 'initial_pos'` in `camera_follow_traj` and similar when accessing `env.viewer`.

**Fixes:**

### 5.1 Make headless configurable
In **iGibson/igibson/scripts/test_flona_gtpos.py**, replace the hardcoded:
```python
headless = False
```
with:
```python
headless = config.get("headless", not os.environ.get("DISPLAY"))
```

In **flodiff/test.yaml**, add:
```yaml
headless: true   # set false if running with a display
```

### 5.2 Guard viewer access
In **iGibson/igibson/scripts/test_flona_gtpos.py**:

- In **`camera_set_and_record`**: wrap viewer updates in `if env.viewer is not None:` before setting `initial_pos`, `initial_view_direction`, and calling `reset_viewer()`.
- In **`camera_follow_traj`**: same — only set `env.viewer.initial_pos`, `initial_view_direction`, and `reset_viewer()` when `env.viewer is not None`.

### 5.3 Qt when running headless
If you run without a display (e.g. SSH, CI), set:
```bash
export QT_QPA_PLATFORM=offscreen
```
before running the script so Qt does not try to use the xcb (X11) plugin.

---

## 6. How to run the evaluation script

From a shell (with flodiff env and iGibson repo available):

```bash
conda activate flodiff
cd /path/to/iGibson

GIBSON_DATASET_PATH=/home/vgmachinist/projects/iGibson/igibson/data/gibson_v2_selected \
QT_QPA_PLATFORM=offscreen \
python -m igibson.scripts.test_flona_gtpos
```

- Use your actual path for `GIBSON_DATASET_PATH` if different.  
- Omit `QT_QPA_PLATFORM=offscreen` if you have a display and want the GUI.  
- In **test.yaml**, set `headless: false` if you run with a display.

---

## Summary of modified files

| Location | Change |
|----------|--------|
| **iGibson** (clone) | `git submodule update --init --recursive` |
| **iGibson/igibson/render/pybind11/include/pybind11/attr.h** | Add `#include <cstdint>` |
| **iGibson/igibson/scripts/test_flona_gtpos.py** | Paths to flodiff and test.yaml; travers map fallback; collision scalar conversion; headless from config; viewer `is not None` guards |
| **flodiff/test.yaml** | `scene_path`; `headless: true` |

All of these were applied so that:
1. iGibson builds with `pip install -e .` in the flodiff conda env (Python 3.8).  
2. The evaluation script runs using flodiff’s **test.yaml** and dataset paths.  
3. The script runs in headless mode without a display and without Qt xcb errors.
