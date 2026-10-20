import contextlib
import os
import shutil
import sys
import tempfile
import types
import unittest

import numpy as np

import registration_tools as rt
from registration_tools import nodes

try:
    from napari_flow_editor import node_transport, plugin_nodes
    HAS_FLOW = True
except ImportError:
    HAS_FLOW = False

try:
    import vt  # noqa: F401
    BACKEND = "vt"
except ImportError:
    try:
        import SimpleITK  # noqa: F401
        BACKEND = "sitk"
    except ImportError:
        BACKEND = None


@contextlib.contextmanager
def flow_context(**context):
    """Stands in for the dispatch context Napari_Flow pushes while a node runs."""
    name = "napari_flow_editor.flow_nodes.decorator"
    previous = sys.modules.get(name)
    stub = types.ModuleType(name)
    stub.get_dispatch_context = lambda: context
    sys.modules[name] = stub
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def stack_registration(frames=3, shape=(20, 30)):
    matrices = np.stack([np.eye(3) for _ in range(frames)])
    matrices[:, 0, 2] = np.arange(frames)
    return {"kind": "affine", "matrix": matrices, "shape": shape, "moving_shape": shape,
            "scale": (2.0, 0.5), "frames": frames}


@unittest.skipUnless(HAS_FLOW, "needs napari_flow_editor")
class TestNapariFlowContract(unittest.TestCase):

    def test_every_node_is_accepted(self):
        discovery = plugin_nodes.discover(modules=["registration_tools"])
        self.assertEqual([(r.name, r.reason) for r in discovery.rejected], [])
        expected = {f"registration_tools:{function.__name__}" for function in nodes.NODES}
        self.assertEqual(set(discovery.nodes), expected)

        sockets = {
            "register_pair": (["reference", "moving"], "vectors"),
            "register_stack": (["stack", "reference"], "vectors"),
            "apply_registration": (["registration", "image"], "image"),
            "apply_registration_labels": (["registration", "labels"], "labels"),
            "registration_to_vectors": (["registration"], "vectors"),
            "save_registration": (["registration"], "vectors"),
            "load_registration": (["image"], "vectors"),
        }
        for name, (inputs, output) in sockets.items():
            with self.subTest(node=name):
                definition = discovery.nodes[f"registration_tools:{name}"]
                self.assertEqual(definition["inputs"], inputs)
                self.assertEqual(list(definition["output_types"].values()), [output])
                self.assertEqual(definition["execution_path"], f"registration_tools.nodes.{name}")
                self.assertEqual(definition["category"], "Plugin: registration-tools")

        parameters = discovery.nodes["registration_tools:register_stack"]["parameters"]
        self.assertEqual(parameters["mode"],
                         {"type": "enum", "default": "reference", "options": ["reference", "consecutive"]})
        self.assertEqual(parameters["transformation"]["options"], list(rt.functional.TRANSFORMATIONS))
        self.assertEqual(parameters["backend"], {"type": "enum", "default": "vt", "options": ["vt", "sitk"]})
        self.assertEqual(parameters["reference_index"], {"type": "int", "default": 0})
        self.assertEqual(parameters["accumulate"], {"type": "bool", "default": True})
        self.assertEqual(parameters["grid_spacing"], {"type": "int", "default": 1})
        apply_parameters = discovery.nodes["registration_tools:apply_registration"]["parameters"]
        self.assertEqual(apply_parameters["fill_value"], {"type": "float", "default": 0.0})
        self.assertEqual(apply_parameters["boundary"]["type"], "enum")

    def test_registration_survives_the_transport(self):
        field = stack_registration()
        field.pop("matrix")
        field.update(kind="field", field=np.random.default_rng(0).random((3, 5, 6, 2)).astype(np.float32))
        folder = tempfile.mkdtemp()
        try:
            for registration in (stack_registration(), field):
                with self.subTest(kind=registration["kind"]):
                    encoded = node_transport.encode((registration, {}), folder)
                    decoded, meta = node_transport.decode(encoded, folder)
                    self.assertEqual(meta, {})
                    key = "matrix" if registration["kind"] == "affine" else "field"
                    np.testing.assert_array_equal(decoded[key], registration[key])
                    image = np.random.default_rng(1).random((3, 20, 30))
                    np.testing.assert_array_equal(rt.apply_registration(image, decoded),
                                                  rt.apply_registration(image, registration))
        finally:
            shutil.rmtree(folder, ignore_errors=True)


class TestNodes(unittest.TestCase):

    def setUp(self):
        self.folder = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.folder, ignore_errors=True)

    def test_dropdowns_offer_the_choices_of_the_functions(self):
        import typing

        for choices, accepted in ((nodes.Transformation, rt.functional.TRANSFORMATIONS),
                                  (nodes.Backend, rt.functional.BACKENDS),
                                  (nodes.StackMode, rt.functional.STACK_MODES),
                                  (nodes.Boundary, rt.functional.BOUNDARY_MODES)):
            self.assertEqual(typing.get_args(choices), accepted)

    def test_apply_nodes(self):
        registration = stack_registration()
        image = np.random.default_rng(0).random((3, 20, 30)).astype(np.float32)
        np.testing.assert_array_equal(nodes.apply_registration(registration, image),
                                      rt.apply_registration(image, registration))
        labels = np.zeros((3, 20, 30), np.int32)
        labels[:, 5:10, 5:15] = 4
        warped = nodes.apply_registration_labels(registration, [labels])  # multiscale list: level 0
        self.assertEqual(warped.dtype, np.int32)
        self.assertEqual(set(np.unique(warped)), {0, 4})

    def test_vectors_node(self):
        arrows, meta = nodes.registration_to_vectors(stack_registration(), spacing=5)
        self.assertEqual(arrows.shape[1:], (2, 3))
        self.assertEqual(meta["layer_type"], "vectors")
        self.assertEqual(meta["scale"], (1.0, 2.0, 0.5))

    def test_save_and_load_with_batch_paths(self):
        registration = stack_registration()
        context = dict(pipeline_dir=self.folder, batch_row={"embryo": "e1/a"}, batch_index=3, node_title="Save")
        with flow_context(**context):
            passed, meta = nodes.save_registration(registration, path="registrations/{embryo}_{row:02d}.npz")
            self.assertIs(passed, registration)
            self.assertTrue(os.path.exists(os.path.join(self.folder, "registrations", "e1_a_03.npz")))
            loaded, _ = nodes.load_registration(path="registrations/{embryo}_{row:02d}")
            np.testing.assert_array_equal(loaded["matrix"], registration["matrix"])
            with self.assertRaisesRegex(ValueError, "embryo"):
                nodes.save_registration(registration, path="{sample}.npz")
        with self.assertRaises(FileExistsError):
            with flow_context(**context):
                nodes.save_registration(registration, path="registrations/{embryo}_{row:02d}", overwrite=False)

    def test_load_checks_the_image(self):
        path = os.path.join(self.folder, "registration.npz")
        rt.save_registration(stack_registration(), path)
        nodes.load_registration(path, image=np.zeros((3, 20, 30)))
        with self.assertRaises(ValueError):
            nodes.load_registration(path, image=np.zeros((4, 20, 30)))
        with self.assertRaises(ValueError):
            nodes.load_registration(path, image=np.zeros((20, 30)))
        with self.assertWarns(UserWarning):
            nodes.load_registration(path, image=np.zeros((3, 20, 31)))

    def test_layer_scale_and_extra_args(self):
        with flow_context(metadata={"axes": "TZYX", "scale": (1.0, 2.0, 0.5, 0.5)}):
            self.assertEqual(nodes._layer_scale(3), (2.0, 0.5, 0.5))
        self.assertIsNone(nodes._layer_scale(3))
        self.assertEqual(nodes._backend_kwargs("vt", " -elastic-sigma 2 2 2 "),
                         {"args_registration": "-elastic-sigma 2 2 2"})
        self.assertEqual(
            nodes._backend_kwargs("sitk", "metric=Correlation optimizer_numberOfIterations=200 "
                                          "pyramid_shrink_factor=4,2,1 optimizer_learningRate=0.5"),
            {"metric": "Correlation", "optimizer_numberOfIterations": 200,
             "pyramid_shrink_factor": (4, 2, 1), "optimizer_learningRate": 0.5},
        )
        with self.assertRaises(ValueError):
            nodes._backend_kwargs("sitk", "Correlation")

    @unittest.skipUnless(BACKEND, "needs vt or SimpleITK")
    def test_register_nodes_use_the_layer_scale(self):
        import scipy.ndimage as ndi

        rng = np.random.default_rng(0)
        reference = ndi.gaussian_filter(rng.random((40, 60, 70)), 2.5).astype(np.float32)
        moving = ndi.shift(reference, (2.0, 3.0, -4.0), order=1, mode="nearest")
        with flow_context(metadata={"scale": (2.0, 0.5, 0.5)}):
            registration, meta = nodes.register_pair(reference, moving, transformation="Translation",
                                                     backend=BACKEND)
        self.assertEqual(meta, {})
        self.assertEqual(registration["scale"], (2.0, 0.5, 0.5))
        np.testing.assert_allclose(registration["matrix"][:-1, -1], (2.0, 3.0, -4.0), atol=0.6)

        stack = np.stack([reference, moving])
        registration, _ = nodes.register_stack(stack, transformation="translation", backend=BACKEND,
                                               use_layer_scale=False)
        self.assertEqual(registration["scale"], (1.0, 1.0, 1.0))
        self.assertEqual(registration["frames"], 2)


if __name__ == "__main__":
    unittest.main()
