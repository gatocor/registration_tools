import inspect
import skimage.filters as skfilters
import skimage.data as skdata
import skimage.exposure as skexposure
import skimage.feature as skfeature
import skimage.morphology as skmorphology

import numpy as np
from skimage import data
import inspect

def infer_output(func):
    sample_image = data.camera()
    sig = inspect.signature(func)
    params = sig.parameters

    # Prepare arguments
    args = {}
    for pname, p in params.items():
        if pname == "image":
            args[pname] = sample_image
        elif p.default is not inspect.Parameter.empty:
            args[pname] = p.default
        else:
            return None  # can't call function safely

    try:
        result = func(**args)
    except Exception:
        return None

    # Classify result
    if isinstance(result, np.ndarray):
        return {"type": "image", "name": "image"}
    elif isinstance(result, (float, int)):
        return {"type": "scalar", "name": "value"}
    elif isinstance(result, tuple):
        if all(isinstance(r, np.ndarray) for r in result):
            return {"type": "tuple[image]", "name": "images"}
        return {"type": "tuple", "name": "result"}
    elif result is None:
        return {"type": "None", "name": "None"}
    else:
        return {"type": type(result).__name__, "name": "result"}

def get_module_functions(funcs, module):

    for name, func in inspect.getmembers(module, inspect.isfunction):
        sig = inspect.signature(func)
        params = sig.parameters
        return_annotation = sig.return_annotation

        # Must have "image" parameter
        if len(params.values()) == 0:
            dtype = "dataset"
            inputs = None
            outputs = ("image",)
        else:

            # Count how many required parameters there are (no default)
            required_params = [
                p for p in params.values()
                if p.default is inspect.Parameter.empty
            ]

            dtype = "function"
            inputs = {"image": {"type": "image", "default": None}}
            if len(required_params) == 0 and [i for i in params.values()][0].name == "image":
                None
            elif len(required_params) == 1 and required_params[0].name == "image":
                None
            # elif name in [i[0] for i in default_map.keys()]:
            #     None
            else:
                # print(f"Skipping {name}: requires multiple parameters or has non-image required parameters ({required_params}). To be implemented.")
                continue

        outputs = infer_output(func)
        funcs[name] = {
            "name": name,
            "function_name": name,
            "type": dtype,
            "inputs": inputs,
            "outputs": outputs,
            "function": func,
            "help": "www.scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters." + name,
            "parameters": {
                p.name: {
                    "type": p.annotation if p.annotation is not inspect._empty else type(p.default),
                    "default": p.default if p.default is not inspect.Parameter.empty else None
                }
                for p in params.values() if p.name != "image"
            }
        }

    return funcs

NODES_SKIMAGE = {}
for module_name, module in [
    ("data", skdata),
    ("filters", skfilters),
    ("exposure", skexposure),
    ("feature", skfeature),
    ("morphology", skmorphology)
]:
    NODES_SKIMAGE[module_name] = {}
    NODES_SKIMAGE[module_name] = get_module_functions(NODES_SKIMAGE[module_name], module)
