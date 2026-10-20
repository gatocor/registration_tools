"""
Runs the example pipelines without the GUI (`nfe run`), binding every Get Layer node to
data/<layer name>.tif.

    python make_example_data.py        # once, writes ./data
    python run_examples.py             # every example
    python run_examples.py 03 08 09    # some of them, by number or file name

Run it in the environment with Napari_Flow and registration_tools, after
`nfe plugins --refresh`. 11_vortex_flow downloads its sample data (~2 MB) the first time;
--offline skips it.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEEDS_DOWNLOAD = {"11_vortex_flow.json"}

# `nfe run --bind` reads files with napari's readers, which a headless run does not
# initialise, so TIFFs are "not readable". Initialising them first is all `nfe` needs.
NFE = ("from napari.plugins import _initialize_plugins; _initialize_plugins(); "
       "from napari_flow_editor.cli import main; raise SystemExit(main())")


def bindings(pipeline, data):
    """--bind arguments for the Get Layer nodes of a pipeline."""
    nodes = json.loads(pipeline.read_text())["nodes"]
    names = sorted({node["parameters"]["layer_name"] for node in nodes if node["type"] == "get_layer"})
    arguments = []
    for name in names:
        path = data / f"{name}.tif"
        if not path.exists():
            raise FileNotFoundError(f"{pipeline.name} needs {path}; run make_example_data.py first.")
        arguments += ["--bind", f"{name}={path}"]
    return arguments


def run(pipeline, data, verbose=False):
    """Runs one pipeline from its own folder (Begin Batch reads batch.csv from there)."""
    command = [sys.executable, "-c", NFE, "--no-regenerate", "run", str(pipeline), *bindings(pipeline, data)]
    started = time.time()
    result = subprocess.run(command, cwd=pipeline.parent, capture_output=not verbose, text=True)
    return result, time.time() - started


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("examples", nargs="*", help="Numbers or file names; all by default.")
    parser.add_argument("--offline", action="store_true", help="Skip the examples that download sample data.")
    parser.add_argument("--folder", type=Path, default=HERE, help="Folder with the pipelines and data/.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show the output of every run.")
    args = parser.parse_args(argv)

    pipelines = sorted(args.folder.glob("[0-9][0-9]_*.json"))
    if args.examples:
        pipelines = [p for p in pipelines if any(p.name == e or p.name.startswith(e) for e in args.examples)]
    if args.offline:
        pipelines = [p for p in pipelines if p.name not in NEEDS_DOWNLOAD]

    failed = []
    for pipeline in pipelines:
        result, seconds = run(pipeline, args.folder / "data", args.verbose)
        ok = result.returncode == 0
        print(f"{'ok' if ok else 'FAILED':6s} {pipeline.name:30s} {seconds:6.1f} s", flush=True)
        if not ok:
            failed.append(pipeline.name)
            if not args.verbose:
                print((result.stdout or "")[-2000:], (result.stderr or "")[-4000:], sep="\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
