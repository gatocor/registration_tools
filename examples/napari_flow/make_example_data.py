"""
Writes the synthetic datasets used by the example pipelines into ./data.

    python make_example_data.py

Every dataset moves a synthetic tissue (nuclei and membranes) with a known motion, and
comes with that motion as a registration (`*_truth.npz`, the same format as Save
Registration writes), so results can be checked with Load Registration or
`registration_tools.load_registration`.

Images are single-channel ImageJ TIFFs (axes and voxel size in their metadata, for Fiji).
napari opens them as layers named after the file, which the pipelines' Get Layer nodes
pick up; the Get Layer nodes declare the axes and pixel size themselves.

drift_2d_nuclei.tif      (T, Y, X) 20 frames, 256x256, pixel 0.5 um. The field of view
drift_2d_membranes.tif   drifts (random walk) and slowly rotates; two channels.
drift_2d_truth.npz       Registration of every frame onto frame 0.
zstack_jitter.tif        (Z, Y, X) 40 slices, 256x256, every slice shifted at random, as in
                         serial sections. The middle slice is not moved.
zstack_clean.tif         The same volume before the shifts.
zstack_jitter_truth.npz  Registration of every slice onto the middle one.
drift_3d.tif             (T, Z, Y, X) 6 time points of a 24x96x96 volume with voxel size
                         (2, 0.5, 0.5) um, drifting, turning and tilting in physical space.
drift_3d_truth.npz       Registration of every time point onto the first one.
"""

import argparse
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import tifffile
from scipy.spatial import cKDTree
from skimage.segmentation import find_boundaries

import registration_tools as rt

HERE = Path(__file__).resolve().parent


def write_tiff(folder, name, data, axes, spacing):
    """ImageJ TIFF with the axes and the voxel size (um) of the last axes in `spacing`."""
    metadata = {"axes": axes.upper(), "unit": "um"}
    if "z" in axes:
        metadata["spacing"] = spacing[axes.index("z")]
    tifffile.imwrite(folder / f"{name}.tif", data, imagej=True, metadata=metadata,
                     resolution=(1 / spacing[-1], 1 / spacing[-2]))


def tissue(shape, spacing, cell_size, seed):
    """Nuclei and membranes of a random tissue, as two float images in [0, 1].

    Cells are the Voronoi regions of random centres, laid out in physical units so that
    anisotropic voxels still give round cells.
    """
    rng = np.random.default_rng(seed)
    spacing = np.asarray(spacing, float)
    extent = np.array(shape) * spacing
    n_cells = max(4, int(np.prod(extent) / cell_size ** len(shape)))
    centres = rng.uniform(0, extent, size=(n_cells, len(shape)))
    positions = np.indices(shape).reshape(len(shape), -1).T * spacing
    distance, owner = cKDTree(centres).query(positions)
    distance, owner = distance.reshape(shape), owner.reshape(shape)

    sigma = 0.6 / spacing * spacing.min()
    membranes = ndi.gaussian_filter(find_boundaries(owner).astype(float), sigma)
    brightness = rng.uniform(0.5, 1.0, n_cells)[owner]
    texture = ndi.gaussian_filter(rng.random(shape), 1.5 * sigma)
    nuclei = ndi.gaussian_filter((distance < 0.3 * cell_size) * brightness, 1.5 * sigma)
    nuclei *= 0.75 + 0.5 * (texture - texture.min()) / np.ptp(texture)
    return nuclei / nuclei.max(), membranes / membranes.max()


def move(image, matrices, margin):
    """Frames of `image` moved by the pull-back `matrices`, given in the coordinates of
    the image cropped by `margin` pixels: frame(matrix @ x) = cropped image(x)."""
    ndim = image.ndim
    offset = np.eye(ndim + 1)
    offset[:ndim, ndim] = margin
    crop = tuple(slice(m, n - m) for m, n in zip(margin, image.shape))
    frames = []
    for matrix in matrices:
        # The same motion on the uncropped canvas, so no empty border comes into view.
        canvas = offset @ matrix @ np.linalg.inv(offset)
        frames.append(ndi.affine_transform(image, np.linalg.inv(canvas), order=3, mode="nearest")[crop])
    return np.stack(frames)


def to_counts(signal, rng, photons=600.0, background=60.0):
    """Shot noise and camera offset, as uint16."""
    return (rng.poisson(background + photons * np.clip(signal, 0, None)) + 100).astype(np.uint16)


def rigid_px(degrees, axes, translation, centre, spacing):
    """Pull-back matrix in pixels of a rotation by `degrees` in the plane `axes` about
    `centre` followed by `translation`, both in physical units (numpy axis order)."""
    ndim = len(spacing)
    rotation = np.eye(ndim)
    a, b = axes
    angle = np.deg2rad(degrees)
    rotation[np.ix_([a, b], [a, b])] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    physical = np.eye(ndim + 1)
    physical[:ndim, :ndim] = rotation
    physical[:ndim, ndim] = centre - rotation @ centre + translation
    scale = np.diag(list(spacing) + [1.0])
    return np.linalg.inv(scale) @ physical @ scale


def truth(matrices, shape, spacing, transformation, reference_index=0):
    return {
        "kind": "affine",
        "matrix": np.stack(matrices),
        "shape": tuple(shape),
        "moving_shape": tuple(shape),
        "scale": tuple(spacing),
        "frames": len(matrices),
        "transformation": transformation,
        "backend": "ground truth",
        "mode": "reference",
        "reference_index": reference_index,
        "accumulate": None,
    }


def drift_2d(folder, rng):
    shape, spacing, margin, frames = (256, 256), (0.5, 0.5), (48, 48), 20
    nuclei, membranes = tissue(tuple(n + 2 * m for n, m in zip(shape, margin)), spacing, 12.0, seed=1)
    centre = (np.array(shape) - 1) / 2 * spacing
    steps = np.cumsum(rng.normal(0, 1.2, size=(frames, 2)), axis=0)  # um
    steps -= steps[0]
    matrices = [rigid_px(0.4 * t, (0, 1), steps[t], centre, spacing) for t in range(frames)]

    for name, image in (("drift_2d_nuclei", nuclei), ("drift_2d_membranes", membranes)):
        stack = to_counts(move(image, matrices, margin), rng)
        write_tiff(folder, name, stack, "tyx", (1.0,) + spacing)
    rt.save_registration(truth(matrices, shape, spacing, "rigid"), folder / "drift_2d_truth.npz")
    return stack.shape


def zstack_jitter(folder, rng):
    shape, spacing, margin = (40, 256, 256), (1.0, 0.5, 0.5), (0, 16, 16)
    nuclei, membranes = tissue(tuple(n + 2 * m for n, m in zip(shape, margin)), spacing, 10.0, seed=2)
    volume = (nuclei + 0.5 * membranes)[:, margin[1]:-margin[1], margin[2]:-margin[2]]
    middle = shape[0] // 2
    shifts = rng.uniform(-8, 8, size=(shape[0], 2))
    shifts[middle] = 0
    matrices = []
    for s in shifts:
        matrix = np.eye(3)
        matrix[:2, 2] = s
        matrices.append(matrix)
    # Each slice is a 2D image; content at x moves to x + shift.
    jittered = np.stack([ndi.shift(plane, s, order=3, mode="nearest") for plane, s in zip(volume, shifts)])

    for name, data in (("zstack_jitter", jittered), ("zstack_clean", volume)):
        write_tiff(folder, name, to_counts(data, rng), "zyx", spacing)
    registration = truth(matrices, shape[1:], spacing[1:], "translation", reference_index=middle)
    registration["mode"] = "consecutive"
    registration["accumulate"] = True
    rt.save_registration(registration, folder / "zstack_jitter_truth.npz")
    return jittered.shape


def drift_3d(folder, rng):
    shape, spacing, margin, frames = (24, 96, 96), (2.0, 0.5, 0.5), (6, 24, 24), 6
    nuclei, _ = tissue(tuple(n + 2 * m for n, m in zip(shape, margin)), spacing, 12.0, seed=3)
    centre = (np.array(shape) - 1) / 2 * np.array(spacing)
    steps = np.cumsum(rng.normal(0, 1.0, size=(frames, 3)), axis=0)  # um
    steps -= steps[0]
    matrices = []
    for t in range(frames):
        turn = rigid_px(2.0 * t, (1, 2), steps[t], centre, spacing)                # about z
        tilt = rigid_px(1.5 * t, (0, 2), np.zeros(3), centre, spacing)            # about y
        matrices.append(turn @ tilt)
    stack = to_counts(move(nuclei, matrices, margin), rng)
    write_tiff(folder, "drift_3d", stack, "tzyx", (1.0,) + spacing)
    rt.save_registration(truth(matrices, shape, spacing, "rigid"), folder / "drift_3d_truth.npz")
    return stack.shape


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=HERE / "data", help="Folder for the datasets.")
    folder = parser.parse_args().output
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for make in (drift_2d, zstack_jitter, drift_3d):
        print(f"{make.__name__:15s} {make(folder, rng)}")
    print(f"written to {folder}")


if __name__ == "__main__":
    main()
