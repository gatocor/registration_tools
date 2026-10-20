import os
import shutil
import tempfile
import unittest
import warnings

import numpy as np
import scipy.ndimage as ndi

import registration_tools as rt

try:
    import vt  # noqa: F401
    HAS_VT = True
except ImportError:
    HAS_VT = False

try:
    import SimpleITK  # noqa: F401
    HAS_SITK = True
except ImportError:
    HAS_SITK = False

BACKENDS = [name for name, available in (("vt", HAS_VT), ("sitk", HAS_SITK)) if available]


def smooth_image(shape, seed=0, sigma=3):
    rng = np.random.default_rng(seed)
    image = ndi.gaussian_filter(rng.random(shape), sigma)
    return ((image - image.min()) / (image.max() - image.min())).astype(np.float32)


def shifted(image, shift):
    """The image with its content moved by `shift` pixels."""
    return ndi.shift(image, shift, order=1, mode="nearest")


def affine_registration(matrix, shape, frames=None):
    return {
        "kind": "affine",
        "matrix": np.asarray(matrix, dtype=float),
        "shape": tuple(shape),
        "moving_shape": tuple(shape),
        "scale": (1.0,) * len(shape),
        "frames": frames,
    }


def translation_matrix(shift):
    matrix = np.eye(len(shift) + 1)
    matrix[:-1, -1] = shift
    return matrix


def rotation_matrix(shape, degrees, shift):
    """Pull-back matrix of a rotation about the centre followed by a shift (2D)."""
    angle = np.deg2rad(degrees)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    centre = (np.array(shape) - 1) / 2
    matrix = np.eye(3)
    matrix[:2, :2] = rotation
    matrix[:2, 2] = centre - rotation @ centre + np.asarray(shift)
    return matrix


def interior(array, margin=8):
    return array[(slice(None),) * (array.ndim - 2) + (slice(margin, -margin),) * 2]


class TestApplyRegistration(unittest.TestCase):

    def test_translation_matches_shift(self):
        image = smooth_image((40, 50))
        shift = np.array([3.0, -5.0])
        out = rt.apply_registration(image, affine_registration(translation_matrix(shift), image.shape))
        np.testing.assert_allclose(interior(out), interior(ndi.shift(image, -shift, order=1)), atol=1e-5)

    def test_field_matches_matrix(self):
        image = smooth_image((20, 30, 40))
        matrix = np.eye(4)
        matrix[:3, :3] = [[1.0, 0.02, 0.0], [-0.03, 1.0, 0.01], [0.0, 0.05, 0.98]]
        matrix[:3, 3] = [1.5, -2.0, 3.0]
        registration = affine_registration(matrix, image.shape)
        # "nearest": with "constant", a sample landing exactly on the image edge flips between
        # the edge and the fill value with float32 rounding.
        expected = rt.apply_registration(image, registration, boundary="nearest")
        for grid_spacing in (1, 3, 7):
            with self.subTest(grid_spacing=grid_spacing):
                field = rt.registration_to_field(registration, grid_spacing=grid_spacing)
                as_field = {key: value for key, value in registration.items() if key != "matrix"}
                as_field.update(kind="field", field=field)
                # Linear interpolation reproduces an affine motion exactly, even on a coarse grid.
                np.testing.assert_allclose(rt.apply_registration(image, as_field, boundary="nearest"),
                                           expected, atol=1e-4)

    def test_labels_keep_their_values(self):
        labels = np.zeros((40, 40), np.uint16)
        labels[5:15, 5:15] = 7
        labels[20:30, 22:35] = 300
        registration = affine_registration(rotation_matrix(labels.shape, 10, (2.5, -3.5)), labels.shape)
        out = rt.apply_registration(labels, registration, order=0)
        self.assertEqual(out.dtype, np.uint16)
        self.assertTrue(set(np.unique(out)) <= {0, 7, 300})
        self.assertEqual(set(np.unique(out)), {0, 7, 300})

    def test_dtypes_are_kept(self):
        registration = affine_registration(translation_matrix([1.5, 0.5]), (30, 30))
        for dtype in (np.uint8, np.uint16, np.float32, np.float64, bool):
            with self.subTest(dtype=dtype):
                image = (smooth_image((30, 30)) * 200).astype(dtype)
                self.assertEqual(rt.apply_registration(image, registration).dtype, np.dtype(dtype))

    def test_stack_with_channels_and_pair_on_stack(self):
        shifts = [(0.0, 0.0), (1.0, -2.0), (3.0, 1.5)]
        stack = np.stack([smooth_image((30, 40), seed=s) for s in range(3)])
        channels = np.stack([stack, 2 * stack], axis=1)  # (T, C, Y, X)
        registration = affine_registration(np.stack([translation_matrix(s) for s in shifts]), (30, 40), frames=3)
        out = rt.apply_registration(channels, registration)
        for t, shift in enumerate(shifts):
            single = rt.apply_registration(stack[t], affine_registration(translation_matrix(shift), (30, 40)))
            np.testing.assert_allclose(out[t, 0], single, atol=1e-6)
            np.testing.assert_allclose(out[t, 1], 2 * single, atol=1e-5)

        pair = affine_registration(translation_matrix((2.0, 2.0)), (30, 40))
        out = rt.apply_registration(stack, pair)
        for t in range(3):
            np.testing.assert_allclose(out[t], rt.apply_registration(stack[t], pair), atol=1e-6)

    def test_output_uses_the_reference_grid(self):
        registration = affine_registration(translation_matrix((1.0, 1.0)), (20, 25))
        registration["moving_shape"] = (32, 32)
        self.assertEqual(rt.apply_registration(smooth_image((32, 32)), registration).shape, (20, 25))

    def test_dask_input_stays_lazy(self):
        import dask.array as da

        stack = np.stack([smooth_image((30, 40), seed=s) for s in range(4)])
        registration = affine_registration(
            np.stack([translation_matrix((t, -t)) for t in range(4)]), (30, 40), frames=4
        )
        lazy = rt.apply_registration(da.from_array(stack, chunks=(2, 15, 20)), registration)
        self.assertIsInstance(lazy, da.Array)
        np.testing.assert_allclose(lazy.compute(), rt.apply_registration(stack, registration), atol=1e-6)

    def test_errors(self):
        stack_registration = affine_registration(np.stack([np.eye(3)] * 3), (30, 40), frames=3)
        with self.assertRaises(ValueError):
            rt.apply_registration(np.zeros((4, 30, 40)), stack_registration)
        with self.assertRaises(TypeError):
            rt.apply_registration(np.zeros((30, 40)), np.zeros((10, 2, 2)))
        with self.assertRaises(ValueError):
            rt.apply_registration(np.zeros((30, 40)), affine_registration(np.eye(3), (30, 40)), boundary="edge")
        with self.assertWarns(UserWarning):
            rt.apply_registration(np.zeros((31, 40)), affine_registration(np.eye(3), (30, 40)))


class TestComposeAndVectors(unittest.TestCase):

    def test_compose_affine(self):
        shape = (40, 50)
        first = affine_registration(rotation_matrix(shape, 4, (1.0, -2.0)), shape)
        second = affine_registration(translation_matrix((2.0, 3.0)), shape)
        composed = rt.compose_registrations(first, second)
        self.assertEqual(composed["kind"], "affine")
        np.testing.assert_allclose(composed["matrix"], first["matrix"] @ second["matrix"])

        image = smooth_image(shape)
        in_two_steps = rt.apply_registration(rt.apply_registration(image, first), second)
        np.testing.assert_allclose(interior(rt.apply_registration(image, composed), 10),
                                   interior(in_two_steps, 10), atol=0.02)

    def test_compose_with_a_field(self):
        shape = (40, 50)
        first = affine_registration(translation_matrix((1.0, -2.0)), shape)
        first_as_field = {key: value for key, value in first.items() if key != "matrix"}
        first_as_field.update(kind="field", field=rt.registration_to_field(first, grid_spacing=4))
        second = affine_registration(rotation_matrix(shape, 3, (2.0, 1.0)), shape)
        composed = rt.compose_registrations(first_as_field, second)
        self.assertEqual(composed["kind"], "field")
        expected = rt.compose_registrations(first, second)
        image = smooth_image(shape)
        np.testing.assert_allclose(rt.apply_registration(image, composed),
                                   rt.apply_registration(image, expected), atol=1e-4)

    def test_vectors_layout(self):
        shape = (40, 50)
        pair = affine_registration(translation_matrix((2.0, -3.0)), shape)
        arrows = rt.registration_to_vectors(pair, spacing=10)
        self.assertEqual(arrows.shape[1:], (2, 2))
        np.testing.assert_allclose(arrows[:, 1], np.tile([2.0, -3.0], (len(arrows), 1)))

        stack = affine_registration(np.stack([translation_matrix((t, 0.0)) for t in range(3)]), shape, frames=3)
        arrows = rt.registration_to_vectors(stack, spacing=10)
        self.assertEqual(arrows.shape[1:], (2, 3))
        self.assertEqual(set(arrows[:, 0, 0]), {0.0, 1.0, 2.0})
        np.testing.assert_allclose(arrows[:, 1, 1], arrows[:, 0, 0])
        np.testing.assert_allclose(arrows[:, 1, 0], 0.0)


class TestSaveLoad(unittest.TestCase):

    def setUp(self):
        self.folder = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.folder, ignore_errors=True)

    def registrations(self):
        shape = (20, 30)
        pair = affine_registration(rotation_matrix(shape, 5, (1.0, 2.0)), shape)
        pair.update(transformation="rigid", backend="vt")
        stack = affine_registration(np.stack([translation_matrix((t, -t)) for t in range(3)]), shape, frames=3)
        stack.update(transformation="translation", backend="sitk", mode="consecutive",
                     reference_index=np.int64(0), accumulate=True)
        field = {key: value for key, value in stack.items() if key != "matrix"}
        field.update(kind="field", field=rt.registration_to_field(stack, grid_spacing=4), transformation="vectorfield")
        return {"pair": pair, "stack": stack, "field": field}

    def test_roundtrip(self):
        image = np.stack([smooth_image((20, 30), seed=s) for s in range(3)])
        for name, registration in self.registrations().items():
            with self.subTest(name=name):
                path = rt.save_registration(registration, os.path.join(self.folder, "sub", name))
                self.assertEqual(path.suffix, ".npz")
                loaded = rt.load_registration(path)
                key = "matrix" if registration["kind"] == "affine" else "field"
                np.testing.assert_array_equal(loaded[key], registration[key])
                for meta in ("kind", "shape", "moving_shape", "scale", "frames", "transformation", "backend"):
                    self.assertEqual(loaded[meta], tuple(registration[meta]) if meta in ("shape", "moving_shape", "scale") else registration[meta])
                target = image if registration["frames"] else image[0]
                np.testing.assert_array_equal(rt.apply_registration(target, loaded),
                                              rt.apply_registration(target, registration))

    def test_suffix_and_overwrite(self):
        registration = self.registrations()["pair"]
        base = os.path.join(self.folder, "registration")
        rt.save_registration(registration, base)
        self.assertTrue(os.path.exists(base + ".npz"))
        rt.load_registration(base)  # suffix is optional when loading too
        rt.save_registration(registration, base)  # overwrite=True by default
        with self.assertRaises(FileExistsError):
            rt.save_registration(registration, base, overwrite=False)
        self.assertEqual(sorted(os.listdir(self.folder)), ["registration.npz"])  # no temporary files left

        # Same permissions as any file the user creates (the umask), not owner-only.
        plain = os.path.join(self.folder, "plain")
        open(plain, "w").close()
        self.assertEqual(os.stat(base + ".npz").st_mode & 0o777, os.stat(plain).st_mode & 0o777)

    def test_foreign_files(self):
        other = os.path.join(self.folder, "other.npz")
        np.savez(other, data=np.zeros(3))
        with self.assertRaises(ValueError):
            rt.load_registration(other)
        with self.assertRaises(FileNotFoundError):
            rt.load_registration(os.path.join(self.folder, "missing.npz"))


@unittest.skipUnless(BACKENDS, "needs vt or SimpleITK")
class TestRegisterPair(unittest.TestCase):

    def test_translation(self):
        cases = [
            ((60, 80), (3.0, -4.0), None),
            ((60, 80), (2.0, 5.0), (2.0, 0.5)),
            ((30, 60, 70), (2.0, 3.0, -4.0), (2.0, 0.5, 0.5)),
        ]
        for backend in BACKENDS:
            for shape, shift, scale in cases:
                with self.subTest(backend=backend, shape=shape, scale=scale):
                    reference = smooth_image(shape, sigma=2.5)
                    registration = rt.register_pair(reference, shifted(reference, shift),
                                                    transformation="translation", backend=backend, scale=scale)
                    self.assertEqual(registration["kind"], "affine")
                    # Pixels of the reference are found `shift` further in the moving image.
                    np.testing.assert_allclose(registration["matrix"][:-1, -1], shift, atol=0.6)
                    np.testing.assert_allclose(registration["matrix"][:-1, :-1], np.eye(len(shape)), atol=1e-6)

    def test_rigid(self):
        from skimage.data import camera

        smooth = smooth_image((120, 140))
        photo = camera().astype(np.float32) / 255
        cases = [
            # (reference, degrees, shift, border of the moving image, corner tolerance in px)
            (smooth, 5, (3.0, -2.0), "nearest", 1.0),
            # Empty corners and a larger motion: SimpleITK's mean squares stopped at 0.5 degrees.
            (photo, 6, (15.0, -10.0), "constant", 1.5),
        ]
        for backend in BACKENDS:
            for reference, degrees, shift, border, tolerance in cases:
                with self.subTest(backend=backend, shape=reference.shape, border=border):
                    truth = rotation_matrix(reference.shape, degrees, shift)
                    moving = ndi.affine_transform(reference, np.linalg.inv(truth), order=1, mode=border)
                    n, m = reference.shape
                    corners = np.array([[0, 0, 1], [n - 1, 0, 1], [0, m - 1, 1], [n - 1, m - 1, 1]], float)
                    registration = rt.register_pair(reference, moving, transformation="rigid", backend=backend)
                    error = np.abs(corners @ registration["matrix"].T - corners @ truth.T).max()
                    self.assertLess(error, tolerance)
                    registered = rt.apply_registration(moving, registration)
                    self.assertLess(np.abs(interior(registered, 40) - interior(reference, 40)).mean(), 0.02)

    def test_vectorfield(self):
        reference = smooth_image((60, 80))
        shift = (2.0, -3.0)
        moving = shifted(reference, shift)
        for backend in BACKENDS:
            for grid_spacing in (1, 4):
                with self.subTest(backend=backend, grid_spacing=grid_spacing):
                    registration = rt.register_pair(reference, moving, transformation="vectorfield",
                                                    backend=backend, grid_spacing=grid_spacing)
                    self.assertEqual(registration["kind"], "field")
                    field = rt.registration_to_field(registration)
                    np.testing.assert_allclose(field[15:45, 20:60].reshape(-1, 2).mean(0), shift, atol=0.5)
                    before = np.abs(interior(moving, 12) - interior(reference, 12)).mean()
                    after = np.abs(interior(rt.apply_registration(moving, registration), 12)
                                   - interior(reference, 12)).mean()
                    self.assertLess(after, before / 3)

    def test_bad_arguments(self):
        image = smooth_image((20, 20))
        with self.assertRaises(ValueError):
            rt.register_pair(image, image, transformation="rotation", backend=BACKENDS[0])
        with self.assertRaises(ValueError):
            rt.register_pair(image, image, backend="elastix")
        with self.assertRaises(ValueError):
            rt.register_pair(image, image[None], backend=BACKENDS[0])
        with self.assertRaises(ValueError):
            rt.register_pair(image, image, backend=BACKENDS[0], scale=(1.0, 1.0, 1.0))


@unittest.skipUnless(BACKENDS, "needs vt or SimpleITK")
class TestRegisterStack(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.drift = np.array([1.5, -2.0])
        cls.base = smooth_image((70, 90), sigma=2.5)
        cls.stack = np.stack([shifted(cls.base, t * cls.drift) for t in range(5)])

    def translations(self, registration):
        return registration["matrix"][:, :-1, -1]

    def test_reference_mode(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                registration = rt.register_stack(self.stack, transformation="translation", backend=backend)
                self.assertEqual(registration["frames"], 5)
                expected = np.arange(5)[:, None] * self.drift
                np.testing.assert_allclose(self.translations(registration), expected, atol=0.6)
                np.testing.assert_array_equal(registration["matrix"][0], np.eye(3))

    def test_consecutive(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                relative = rt.register_stack(self.stack, mode="consecutive", accumulate=False,
                                             transformation="translation", backend=backend)
                expected = np.vstack([[0.0, 0.0], np.tile(self.drift, (4, 1))])
                np.testing.assert_allclose(self.translations(relative), expected, atol=0.6)

                accumulated = rt.register_stack(self.stack, mode="consecutive", transformation="translation",
                                                backend=backend)
                # Chaining is exact; the backend's per-pair error adds up along the chain.
                np.testing.assert_allclose(self.translations(accumulated),
                                           np.cumsum(self.translations(relative), axis=0), atol=1e-6)
                error = np.abs(self.translations(accumulated) - np.arange(5)[:, None] * self.drift).max(1)
                self.assertTrue(np.all(error <= 0.6 * np.arange(5) + 1e-6), error)

    def test_middle_reference(self):
        backend = BACKENDS[0]
        expected = (np.arange(5) - 2)[:, None] * self.drift
        for mode in ("reference", "consecutive"):
            with self.subTest(mode=mode):
                registration = rt.register_stack(self.stack, mode=mode, reference_index=-3,
                                                 transformation="translation", backend=backend)
                self.assertEqual(registration["reference_index"], 2)
                np.testing.assert_allclose(self.translations(registration), expected, atol=1.0)
        relative = rt.register_stack(self.stack, mode="consecutive", reference_index=2, accumulate=False,
                                     transformation="translation", backend=backend)
        # Frames before the reference are registered onto the next frame.
        np.testing.assert_allclose(self.translations(relative)[0], -self.drift, atol=0.6)
        np.testing.assert_allclose(self.translations(relative)[4], self.drift, atol=0.6)

    def test_external_reference(self):
        backend = BACKENDS[0]
        reference = shifted(self.base, -self.drift)
        registration = rt.register_stack(self.stack, reference, transformation="translation", backend=backend)
        np.testing.assert_allclose(self.translations(registration), (np.arange(5) + 1)[:, None] * self.drift,
                                   atol=0.6)
        with self.assertRaises(ValueError):
            rt.register_stack(self.stack, reference, mode="consecutive", backend=backend)

    def test_vectorfield_accumulates(self):
        backend = BACKENDS[0]
        registration = rt.register_stack(self.stack, mode="consecutive", transformation="vectorfield",
                                         backend=backend, grid_spacing=4)
        self.assertEqual(registration["kind"], "field")
        self.assertEqual(registration["field"].shape[0], 5)
        field = rt.registration_to_field(registration)
        means = field[:, 20:50, 25:65].reshape(5, -1, 2).mean(1)
        np.testing.assert_allclose(means, np.arange(5)[:, None] * self.drift, atol=1.0)
        aligned = rt.apply_registration(self.stack, registration)
        error = np.abs(interior(aligned, 15) - interior(self.stack[:1], 15)).mean()
        self.assertLess(error, np.abs(interior(self.stack, 15) - interior(self.stack[:1], 15)).mean() / 3)

    def test_slices_of_a_z_stack(self):
        # A (Z, Y, X) stack is aligned slice by slice with 2D registrations.
        registration = rt.register_stack(self.stack, mode="consecutive", transformation="rigid",
                                         backend=BACKENDS[0])
        self.assertEqual(registration["matrix"].shape, (5, 3, 3))
        self.assertEqual(registration["shape"], (70, 90))


if __name__ == "__main__":
    warnings.simplefilter("ignore", category=UserWarning)
    unittest.main()
