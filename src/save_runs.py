from pathlib import Path
import json
import numpy as np
from datetime import datetime


def save_run_results(
    folder: str,
    filename: str,
    J: float,
    KI: float,
    cfg,
    extra: dict | None = None
):
    """
    Saves fracture run results to JSON file.

    Parameters
    ----------
    folder : str
        Directory to save into.
    filename : str
        Name of file (without extension).
    J : float
    KI : float
    cfg : Config object
    extra : optional dictionary for additional data
    """

    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)

    file_path = folder_path / f"{filename}.json"

    data = {
        "timestamp": datetime.now().isoformat(),
        "J_N_per_m": float(J),
        "KI_Pa_sqrt_m": float(KI),
        "material": {
            "E": cfg.E,
            "nu": cfg.nu,
            "plane_stress": cfg.plane_stress,
            "thickness": cfg.thickness,
        },
        "geometry": {
            "W": cfg.W,
            "H": cfg.H,
            "a": cfg.a,
        },
        "loading": {
            "traction": cfg.traction,
        }
    }

    if extra is not None:
        data["extra"] = extra

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Saved results to: {file_path}")