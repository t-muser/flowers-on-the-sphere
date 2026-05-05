"""Small MITgcm MDS metadata helpers shared by experiments."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np


def parse_mds_meta(path: Path) -> dict[str, Any]:
    """Parse the subset of MITgcm MDS metadata needed by the datagen readers."""
    text = Path(path).read_text()

    def _int_value(name: str, default: int | None = None) -> int | None:
        match = re.search(rf"{name}\s*=\s*\[\s*([0-9]+)\s*\]", text)
        if match is None:
            if default is None:
                raise ValueError(f"{path}: missing {name}")
            return default
        return int(match.group(1))

    dim_match = re.search(r"dimList\s*=\s*\[(.*?)\];", text, re.S)
    if dim_match is None:
        raise ValueError(f"{path}: missing dimList")
    dim_nums = [int(v) for v in re.findall(r"[-+]?\d+", dim_match.group(1))]
    if len(dim_nums) % 3 != 0:
        raise ValueError(f"{path}: malformed dimList")
    dim_list = [
        (dim_nums[i], dim_nums[i + 1], dim_nums[i + 2])
        for i in range(0, len(dim_nums), 3)
    ]

    fields_match = re.search(r"fldList\s*=\s*\{(.*?)\};", text, re.S)
    fields = []
    if fields_match is not None:
        fields = [
            field.strip()
            for field in re.findall(r"'([^']+)'", fields_match.group(1))
        ]

    prec_match = re.search(r"dataprec\s*=\s*\[\s*'([^']+)'\s*\]", text)
    dataprec = prec_match.group(1).strip() if prec_match else "float32"

    return {
        "n_dims": _int_value("nDims"),
        "dim_list": dim_list,
        "dataprec": dataprec,
        "nrecords": _int_value("nrecords"),
        # Static grid MDS files (Depth, hFacC, XC, ...) have no timeStepNumber.
        "time_step": _int_value("timeStepNumber", default=-1),
        "fields": fields,
    }


def mds_dtype(dataprec: str) -> np.dtype:
    """Return the big-endian NumPy dtype for an MDS ``dataprec`` value."""
    if dataprec == "float32":
        return np.dtype(">f4")
    if dataprec == "float64":
        return np.dtype(">f8")
    raise ValueError(f"Unsupported MDS dataprec: {dataprec}")

