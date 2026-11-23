"""The categorical dataset colours pass the palette checks they are drawn under."""

from ne_aligned_utils import config
from ne_aligned_utils.paper.rendering import styles
from scripts.figures.palette_validation import validate


def test_the_dataset_palette_covers_every_dataset_in_slot_order():
    assert list(styles.DATASET_COLORS) == [d for d in config.DATASET_LABELS if d != "mnist"]
    assert len(set(styles.DATASET_COLORS.values())) == len(styles.DATASET_COLORS)


def test_the_dataset_palette_passes_the_categorical_checks_on_white():
    """Run, not remembered. The adjacent pairlist is the one a stacked bar actually needs.

    Contrast returns RELIEF rather than PASS by design -- three slots sit below 3:1 on white -- and
    the relief is discharged by the direct labels every bar carries.
    """
    rows, ok = validate(list(styles.DATASET_COLORS.values()), surface="#ffffff", pairs="adjacent")
    states = {name: state for name, state, _ in rows}
    assert ok, rows
    assert states["Lightness band"] == "PASS"
    assert states["Chroma floor"] == "PASS"
    assert states["CVD separation"] == "PASS"
    assert states["Normal-vision floor"] == "PASS"
    assert states["Contrast vs surface"] == "RELIEF"
