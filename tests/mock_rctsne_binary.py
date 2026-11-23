"""Emulate the rc-tsne binary's .dat contract without a compiled executable.

Accept version, input, output, and thread arguments. Parse the real header for
alpha/beta assertions; offsets assume nonzero alpha and perplexity without optional
lists. NE_ALIGNED_MOCK_RCTSNE_* switches control header dumps, failures, missing
output, dimensions, and non-finite coordinates.
"""

import json
import os
import struct
import sys


def main() -> int:
    _version, infile, outfile, _nthreads = sys.argv[1:5]

    with open(infile, "rb") as handle:
        blob = handle.read()

    n, d, num_label_vals = struct.unpack_from("=iii", blob, 0)
    alpha, beta = struct.unpack_from("=dd", blob, 12)
    labels = list(struct.unpack_from(f"={n}i", blob, 28))
    theta, perplexity = struct.unpack_from("=dd", blob, 28 + 4 * n)
    map_dims, max_iter = struct.unpack_from("=ii", blob, 44 + 4 * n)

    dump = os.environ.get("NE_ALIGNED_MOCK_RCTSNE_DUMP")
    if dump:
        with open(dump, "w") as handle:
            json.dump(
                {
                    "n": n,
                    "d": d,
                    "num_label_vals": num_label_vals,
                    # `alpha` is the paper's beta and `beta` the paper's alpha. Recorded under the
                    # names the file uses, so a test asserting the mapping reads the raw truth.
                    "alpha": alpha,
                    "beta": beta,
                    "labels": labels,
                    "theta": theta,
                    "perplexity": perplexity,
                    "map_dims": map_dims,
                    "max_iter": max_iter,
                },
                handle,
            )

    if os.environ.get("NE_ALIGNED_MOCK_RCTSNE_FAIL"):
        sys.stderr.write("mock rc-tsne binary asked to fail\n")
        return 3
    if os.environ.get("NE_ALIGNED_MOCK_RCTSNE_SILENT"):
        return 0

    rows = int(os.environ.get("NE_ALIGNED_MOCK_RCTSNE_ROWS", n))
    width = int(os.environ.get("NE_ALIGNED_MOCK_RCTSNE_WIDTH", map_dims))

    # Deterministic and not degenerate: the row index against its label, so a test can tell the
    # two batches apart and confirm row order survived the round trip.
    coords = []
    for i in range(rows):
        label = labels[i] if i < len(labels) else 0
        coords.extend([float(i)] + [float(label)] * (width - 1))
    if os.environ.get("NE_ALIGNED_MOCK_RCTSNE_NAN") and coords:
        coords[0] = float("nan")

    with open(outfile, "wb") as handle:
        handle.write(struct.pack("=i", rows))
        handle.write(struct.pack("=i", width))
        handle.write(struct.pack(f"={len(coords)}d", *coords))
        handle.write(struct.pack("=i", max_iter))
        handle.write(struct.pack(f"={max_iter}d", *([1.0] * max_iter)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
