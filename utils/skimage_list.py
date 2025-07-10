import inspect
import skimage.filters as skfilters
import skimage.data as skdata
import skimage.draw as skdraw
import skimage.exposure as skexposure
import skimage.feature as skfeature
import skimage.morphology as skmorphology
import skimage.restoration as skrestoration
import skimage.segmentation as sksegmentation
import skimage.transform as sktransform

import numpy as np
from skimage import data
import inspect
import json

def parse_return_from_doc(func):
    doc = inspect.getdoc(func)
    if not doc:
        return None

    # Try to extract the "Returns" section
    lines = doc.splitlines()
    returns_section = []
    in_returns = False

    for line in lines:
        if line.strip().lower() == "returns":
            in_returns = True
            continue
        if in_returns:
            if line.strip() == "" or re.match(r"^\w", line):  # blank or new section
                break
            returns_section.append(line.strip())

    if not returns_section:
        return None

    # Join and interpret
    return_text = " ".join(returns_section).lower()

def classify_output(obj):
    if isinstance(obj, np.ndarray):
        return {"type": "image", "name": "image"}
    elif isinstance(obj, (int, float, np.integer, np.floating)):
        return {"type": "scalar", "name": "value"}
    elif obj is None:
        return {"type": "None", "name": "None"}
    else:
        return {"type": type(obj).__name__, "name": "result"}

def infer_output(func):
    sample_image = data.coins()
    sig = inspect.signature(func)
    params = sig.parameters

    args = {}
    for pname, p in params.items():
        if pname == "image":
            args[pname] = sample_image
        elif p.default is not inspect.Parameter.empty:
            args[pname] = p.default
        else:
            return None  # can't safely call function

    try:
        result = func(**args)
    except Exception as e:
        print(f"Error calling function {func.__name__}: {e}")
        return None

    if isinstance(result, tuple):
        return {str(i): j for i,j in enumerate(tuple(classify_output(r) for r in result))}
    else:
        return {"0":classify_output(result)}

def get_module_functions(funcs, module, module_name):

    for name, func in inspect.getmembers(module, inspect.isfunction):
        sig = inspect.signature(func)
        params = sig.parameters
        return_annotation = sig.return_annotation

        # Must have "image" parameter
        if len(params.values()) == 0:
            dtype = "image"
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
                print("Done ", name, required_params)
                None
            elif len(required_params) == 1 and required_params[0].name == "image":
                print("Done ", name, required_params)
                None
            # elif name in [i[0] for i in default_map.keys()]:
            #     None
            else:
                print("Fail ", name, required_params)
                # print(f"Skipping {name}: requires multiple parameters or has non-image required parameters ({required_params}). To be implemented.")
                # continue

        outputs = None#infer_output(func)
        funcs[name] = {
            "name": name,
            "function_name": name,
            "type": dtype,
            "inputs": {},
            "outputs": outputs,
            "function": f"{module_name}.{name}",
            "help": f"https://scikit-image.org/docs/stable/api/skimage.{module_name}.html#skimage.{module_name}.{name}",
            "parameters": {
                p.name: {
                    "type": p.annotation if p.annotation is not inspect.Parameter.empty else type(p.default).__name__,
                    "default": p.default if p.default is not inspect.Parameter.empty else None
                }
                for p in params.values()
            }
        }

    return funcs

NODES_SKIMAGE = {}
for module_name, module in [
    # ("data", skdata),
    ("draw", skdraw),
    # ("filters", skfilters),
    # ("exposure", skexposure),
    # ("feature", skfeature),
    # ("morphology", skmorphology)
    # ("restoration", skrestoration),
    # ("segmentation", sksegmentation),
    # ("transform", sktransform),
]:
    NODES_SKIMAGE[module_name] = {}
    NODES_SKIMAGE[module_name] = get_module_functions(NODES_SKIMAGE[module_name], module, module_name)

with open("skimage_nodes.json", "w") as f:
    json.dump(NODES_SKIMAGE, f, default=str, indent=2)