"""Plotting helpers."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STRUCT_NAMES = {
    1: "LV", 2: "RV", 3: "LA", 4: "RA",
    5: "Aorta", 6: "PA", 7: "SVC", 8: "IVC",
}


def save_fig(path, dpi: int = 200) -> None:
    """Save figure with tight layout."""
    path = __import__("pathlib").Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
