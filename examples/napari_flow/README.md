# Napari_Flow examples

Pipelines that exercise the registration_tools nodes in [Napari_Flow](https://git.embl.org/grp-torres-sanchez/Napari_Flow), with synthetic data whose true motion is known.

## Setup

registration_tools lives in its own environment (e.g. `napari_flow-registration`, see the main README):

```bash
nfe plugins --refresh --env napari_flow-registration   # once, from the editor's environment
cd examples/napari_flow
conda run -n napari_flow-registration python make_example_data.py   # writes data/ (about 20 MB)
```

The registration nodes then run in `napari_flow-registration` and everything else in the editor's environment. Everything also works with the editor running in `napari_flow-registration` itself (after `nfe plugins --refresh` there), and a little faster: no node then has to start a separate process.

`make_example_data.py` writes TIFFs of a synthetic tissue (nuclei and membranes) moved by a known motion. Each comes with that motion as a registration (`data/*_truth.npz`), the format Save Registration writes. You can load it with Load Registration (or `registration_tools.load_registration`) to compare against what the nodes find.

## Running them

In napari (`napari`, then **Plugins › Napari Flow**):

1. For the examples that use `data/` (03–06, 08, 09), open the files they need in napari first (**File › Open** or drag and drop). Their Get Layer nodes find the layers by file name.
2. **File → Load Pipeline**, pick the `.json`, and **Run**.

Without the GUI:

```bash
python run_examples.py            # all of them, with the editor's environment's Python
python run_examples.py 03 08 09   # some, by number
```

`run_examples.py` binds every Get Layer node to `data/<layer name>.tif`. Use it instead of a bare `nfe run --bind`, which cannot read TIFFs in a headless run (see Notes).

## The examples

| | Pipeline | What it shows | Data |
|---|---|---|---|
| 01 | `01_pair_rigid.json` | Register Pair, rigid, with vt and with SimpleITK; Apply Registration; Compare Images; arrows | `camera` (built in) rotated 6° and shifted |
| 02 | `02_pair_vectorfield.json` | Register Pair, vectorfield on a coarse grid (`grid_spacing` 4), vt and SimpleITK | `camera` (built in) with a swirl |
| 03 | `03_drift_correction.json` | Register Stack (consecutive, accumulated) on one channel, applied to both channels and to labels (Apply Registration Labels); compared with the ground truth | `drift_2d_nuclei`, `drift_2d_membranes` |
| 04 | `04_stack_modes.json` | The Register Stack modes side by side: onto frame 0, chained onto the last frame (`reference_index` -1), frame to frame (`accumulate` off) | `drift_2d_nuclei` |
| 05 | `05_zstack_alignment.json` | Slice alignment of a z-stack (consecutive translation from the middle slice); compared with the clean volume | `zstack_jitter`, `zstack_clean` |
| 06 | `06_3d_voxel_size.json` | 3D rigid registration of a time-lapse with anisotropic voxels, with and without `use_layer_scale` | `drift_3d` |
| 07 | `07_motion_field.json` | Frame-to-frame motion as a vectorfield (`accumulate` off), drawn as arrows | `moving_nuclei` (built in) |
| 08 | `08_save_registration.json` | Save Registration (to `registrations/drift_2d.npz`, next to the pipeline) | `drift_2d_nuclei` |
| 09 | `09_reuse_registration.json` | Load Registration from 08 and apply it to both channels (run 08 first) | `drift_2d_nuclei`, `drift_2d_membranes` |
| 10 | `10_batch_registration.json` | Begin Batch over `batch.csv` (CSV columns fill Sample Image and Shift), one file per row: `registrations/batch_{dataset}.npz` | built-in samples |
| 11 | `11_vortex_flow.json` | Vectorfield of a real particle-image-velocimetry pair, next to Napari_Flow's Optical Flow | `vortex` (downloaded once, ~2 MB) |

In the viewer, the reference is magenta and the moving or registered images are green, with additive blending, so aligned structures look white. Some layers start hidden; toggle them to compare. Arrows run from where the content is in the reference to where it is in the moving image. For stacks there is one set per frame, so move the time (or z) slider.

## What to expect

Compare Images puts its table in the table dashboard. These SSIM values (1 = identical) come from `python run_examples.py` with vt-python 2.0 and SimpleITK 2.5:

| Example | Before | After |
|---|---|---|
| 01, rigid | moving 0.42 | vt 0.88, SimpleITK 0.84 |
| 02, vectorfield | swirled 0.80 | vt 0.94, SimpleITK 0.92 |
| 03, drift correction | | 0.99 against the ground truth |
| 05, z-stack | jittered 0.21 | aligned 0.92 against the clean volume |
| 06, 3D | | voxel size 0.99, pixels 0.79 against the ground truth |

The whole-image values of 01 and 02 include the empty borders that rotation and shifts bring in. Example 06 is the reason `use_layer_scale` is on by default: with 2 µm slices and 0.5 µm pixels, registering in pixels gets the tilt wrong.

## Notes

- **Batch (10):** Begin Batch opens `batch.csv` relative to the folder Napari_Flow was started from. Start napari (or `run_examples.py`) from this folder, or pick the CSV again in the Begin Batch node.
- **Headless TIFFs:** a headless `nfe run --bind name=file.tif` fails with "No compatible readers", because it does not initialise napari's readers. `run_examples.py` initialises them before calling `nfe`.
- **Generated files:** `data/` and `registrations/` are generated and not tracked by git. Re-run `make_example_data.py` to recreate the data; it is deterministic.
