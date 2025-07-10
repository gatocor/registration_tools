def list_files(folder: str, file_pattern: str = "*.tif"):
    """
    List files in a folder matching a pattern.
    
    Parameters:
        folder (str): The folder to search in.
        file_pattern (str): The pattern to match files against (default: "*.tif").
    
    Returns:
        list: A list of file paths matching the pattern.
    """
    import glob
    import os
    
    if not os.path.isdir(folder):
        raise ValueError(f"Folder does not exist: {folder}")
    
    return glob.glob(os.path.join(folder, file_pattern))

NODES_DATA = {}
NODES_DATA["data"] = {}
NODES_DATA["data"]["data_loader"] = {
    "name": "loader",
    "type": "function",
    "inputs": None,
    "outputs": ("image",),
    "function": list_files,
    "help": "",
    "parameters": {
        "folder": { 
            "type": "folder",
            "default": "",
        },
        "file_pattern": { 
            "type": str,
            "default": "*.tif",
        }
    }
}
