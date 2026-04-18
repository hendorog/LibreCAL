#!/usr/bin/env python3

"""
Converts LibreCAL Touchstone files to a format that a Siglent VNA can use.

Usage:

$ python3 convert_siglent.py [--ports 2|4] /Volumes/LIBRECAL_R /Volumes/LIBRECAL_RW

(or your paths or drive letters, as appropriate)

This will create a "siglent" directory in your LIBRECAL_RW volume.  Once
the files are present the LibreCAL firmware enters Siglent emulation
automatically when it is next powered on (hold the FUNCTION button at
power-on to force it back into the default LibreCAL mode).  The instrument
will then calibrate your VNA as if you were using a SEM5032A (2-port,
default) or SEM5004A (4-port).

By default the script emits the 2-port SEM5032A layout.  This matches a
genuine SEM5032A memory dump exactly (CSV columns, info.dat header bytes,
keyword names and padding) and is the layout that has been observed to work
reliably on SNA5000A series instruments.  Use --ports 4 to get the original
4-port layout.

Tested on SVA1032X and SNA5000A.
"""

__author__ = "Joshua Wise"
__copyright__ = "Copyright (c) 2025 Accelerated Tech, Inc."
__license__ = "MIT"

import argparse
import io
from itertools import combinations
from pathlib import Path
import sys
import numpy as np
from zipfile import ZipFile, ZIP_DEFLATED
import struct
import hashlib
from time import strftime, gmtime

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("indir", type=Path, help="read-only LibreCAL volume")
parser.add_argument("outdir", type=Path, help="read/write LibreCAL volume")
parser.add_argument("--ports", type=int, default=2, choices=(2, 4),
                    help="number of ports to emulate (default 2 -> SEM5032A)")
parser.add_argument("--modules", type=int, default=1,
                    help="number of module entries (and data<N>.zip files) to "
                         "emit (default 1).  Genuine Siglent eCals expose "
                         "multiple selectable modules in the VNA GUI; all "
                         "modules we emit carry identical LibreCAL data.")
parser.add_argument("--factory2-dir", type=Path, default=None,
                    help="directory with alternate per-port s1p files to use "
                         "for the second module (Factory2).  Implies "
                         "--modules 2.  File names matching "
                         "'librecal_p<n>_<open|short|load>_allterm.s1p' or "
                         "any of the LibreCAL/Jupyter-notebook conventions "
                         "are accepted.  THROUGH s-parameters are taken from "
                         "indir since Factory2 typically reuses them.")
args = parser.parse_args()

indir = args.indir
outdir = args.outdir
NPORTS = args.ports
if args.factory2_dir is not None and args.modules < 2:
    args.modules = 2

PORT_NUMS = list(range(1, NPORTS + 1))
PORT_LETTERS = [chr(ord('A') + i) for i in range(NPORTS)]
PAIRS = list(combinations(range(NPORTS), 2))  # index pairs into PORT_NUMS/PORT_LETTERS

FREQ_UNIT = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}


def say_open(name, *a):
    print(f"reading {name}", file=sys.stderr)
    return open(name, *a)


def _parse_touchstone_header(line):
    """Parse a '# Hz S MA R 50' style option line; returns (freq_scale, fmt)
    where fmt is 'RI'/'MA'/'DB'."""
    parts = line.lstrip("#").strip().split()
    # touchstone option order is <freq_unit> <param> <format> R <ref>
    unit = parts[0].lower() if parts else "ghz"
    fmt = (parts[2].upper() if len(parts) >= 3 else "RI")
    return FREQ_UNIT.get(unit, 1e9), fmt


def _ab_to_ri(a, b, fmt):
    """Convert a touchstone (a, b) pair in the given fmt to a complex
    reflection/transmission coefficient (returned as (re, im))."""
    if fmt == "RI":
        return a, b
    if fmt == "MA":
        mag = a
    elif fmt == "DB":
        mag = 10.0 ** (a / 20.0)
    else:
        raise ValueError(f"unsupported touchstone format {fmt!r}")
    rad = b * np.pi / 180.0
    return mag * np.cos(rad), mag * np.sin(rad)


def _read_touchstone(path, n_params):
    """Read a touchstone file with `n_params` complex s-parameters per row.
    Returns (freqs_hz, [re0, im0, re1, im1, ..., re<n-1>, im<n-1>])."""
    freq_scale = 1e9  # default GHz if no option line
    fmt = "RI"
    freqs = []
    out = [[] for _ in range(2 * n_params)]
    with say_open(path, "r") as f:
        for l in f:
            l = l.strip()
            if not l or l.startswith("!"):
                continue
            if l.startswith("#"):
                freq_scale, fmt = _parse_touchstone_header(l)
                continue
            parts = [float(x) for x in l.split()]
            freqs.append(parts[0] * freq_scale)
            for k in range(n_params):
                r, i = _ab_to_ri(parts[1 + 2 * k], parts[2 + 2 * k], fmt)
                out[2 * k].append(r)
                out[2 * k + 1].append(i)
    return freqs, out


def read_s1p(path):
    freqs, cols = _read_touchstone(path, 1)
    return freqs, cols[0], cols[1]


def read_s2p(path):
    return _read_touchstone(path, 4)


# Accepted file-name patterns for per-port standards.  The first match in
# the given directory wins.  {port} is the 1-based port number, {std} is
# lowercase, {STD} is uppercase.
STD_PATTERNS = [
    "P{port}_{STD}.s1p",
    "librecal_p{port}_{std}_allterm.s1p",
    "ecal_{std}_p{port}.s1p",
]
THRU_PATTERNS = [
    "P{a}{b}_THROUGH.s2p",
    "librecal_p{a}_p{b}_thru.s2p",
    "ecal_thru_p{a}_p{b}.s2p",
]


def find_std_file(directory, port, std):
    for pat in STD_PATTERNS:
        p = directory / pat.format(port=port, std=std.lower(), STD=std.upper())
        if p.exists():
            return p
    return None


def find_thru_file(directory, port_a, port_b):
    for pat in THRU_PATTERNS:
        p = directory / pat.format(a=port_a, b=port_b)
        if p.exists():
            return p
    return None


def build_axes(std_dir, thru_fallback_dir=None):
    """Read all per-port standards + THROUGHs from the supplied directories
    and return (freqs_hz, axes) with the 33/129-column Siglent layout.

    std_dir: directory searched for per-port OPEN/SHORT/LOAD.
    thru_fallback_dir: if a THROUGH file is not found in std_dir, fall back
    here.  Only used when std_dir is not indir.
    """
    freqs = []
    axes = [freqs]

    # Per-port one-port standards.  LibreCAL has no physical ATT standard;
    # the firmware does not switch state when asked for per-port ATT, so
    # the physical state during an ATT request is whatever was last
    # selected (typically LOAD).  Storing LOAD values in the ATT slot
    # keeps stored and measured values consistent.
    for port in PORT_NUMS:
        load_re, load_im = None, None
        for std in ("OPEN", "SHORT", "LOAD"):
            p = find_std_file(std_dir, port, std)
            if p is None and thru_fallback_dir is not None:
                p = find_std_file(thru_fallback_dir, port, std)
            if p is None:
                raise FileNotFoundError(
                    f"no {std} file for port {port} under {std_dir}")
            fr, re_, im_ = read_s1p(p)
            if not freqs:
                freqs[:] = fr
            axes.append(re_)
            axes.append(im_)
            if std == "LOAD":
                load_re, load_im = re_, im_
        axes.append(list(load_re))
        axes.append(list(load_im))

    # THROUGH s-parameters for every port pair.  When the THRU comes from
    # thru_fallback_dir (i.e. a different freq grid than the per-port
    # files), interpolate onto the per-port grid so we don't have to
    # truncate the whole module to the intersection of the two grids.
    through_cache = {}
    for i, j in PAIRS:
        a, b = PORT_NUMS[i], PORT_NUMS[j]
        p = find_thru_file(std_dir, a, b)
        using_fallback = False
        if p is None and thru_fallback_dir is not None:
            p = find_thru_file(thru_fallback_dir, a, b)
            using_fallback = True
        if p is None:
            raise FileNotFoundError(
                f"no THROUGH for ports {a}-{b} under {std_dir}")
        thru_freqs, cols = read_s2p(p)
        if using_fallback and (len(thru_freqs) != len(freqs)
                               or thru_freqs[0] != freqs[0]
                               or thru_freqs[-1] != freqs[-1]):
            # Linear interpolation of each (re, im) column onto the
            # per-port grid.  Freq points outside the thru range are
            # clamped to the endpoints by np.interp, which is fine for
            # the smooth S-parameters of an internal through path.
            xp = np.asarray(thru_freqs)
            x = np.asarray(freqs)
            cols = [list(np.interp(x, xp, np.asarray(c))) for c in cols]
            print(f"  (interpolated THRU {a}-{b} from "
                  f"{len(thru_freqs)} -> {len(freqs)} points)")
        through_cache[(i, j)] = cols
        for c in cols:
            axes.append(c)

    # CF = THROUGH again (firmware redirects 'SL ATT,n,m' to the THRU
    # switch state, so the physical hardware presents the THROUGH
    # response in the CF slot).
    for i, j in PAIRS:
        for c in through_cache[(i, j)]:
            axes.append(c)

    axes[0] = freqs

    # Truncate all columns to the shortest (in case any file has fewer
    # points, e.g. early LibreCALs with truncated P34_THROUGH.s2p).
    shortest = min(len(a) for a in axes)
    axes = [a[:shortest] for a in axes]
    return freqs[:shortest], axes


# Build the CSV header row once (depends on NPORTS, not the module).
pair_labels = [PORT_LETTERS[i] + PORT_LETTERS[j] for (i, j) in PAIRS]
csv_header_cols = (["#freq"] + PORT_LETTERS
                   + [f"T_{p}" for p in pair_labels]
                   + [f"CF_{p}" for p in pair_labels]
                   + ["END"])

info_txt_lines = open(indir / "info.txt", "r").read().splitlines()


def build_module_zip(module_name, axes):
    """Produce a zip blob whose single entry is '<module_name>.csv'
    carrying the Siglent-formatted cal data."""
    buf = io.BytesIO()
    ar = np.vstack(axes, dtype=np.float64).transpose()
    with ZipFile(buf, "w", compression=ZIP_DEFLATED, compresslevel=9) as zf, \
         zf.open(f"{module_name}.csv", "w") as outf_b, \
         io.TextIOWrapper(outf_b, newline='\x0A') as outf:
        for l in info_txt_lines:
            outf.write(f"! {l.strip()}\n")
        outf.write(",".join(csv_header_cols) + "\n")
        np.savetxt(outf, ar, delimiter=",")
    return buf.getvalue()


# Assemble the axes for every requested module.
module_names = []
module_data = []  # list of (freqs, axes) per module
for n in range(args.modules):
    if n == 0:
        name = "Factory"
        freqs, axes = build_axes(indir)
    elif n == 1 and args.factory2_dir is not None:
        name = "Factory2"
        freqs, axes = build_axes(args.factory2_dir, thru_fallback_dir=indir)
    else:
        # Extra modules without a dedicated data source reuse the
        # Factory axes (useful for exercising multi-module parsing).
        name = "Factory" if n == 0 else f"Factory{n + 1}"
        freqs, axes = module_data[0]
    module_names.append(name)
    module_data.append((freqs, axes))
    print(f"module '{name}': {len(freqs)} freq points, "
          f"{int(freqs[0])}..{int(freqs[-1])} Hz")

module_zips = [build_module_zip(name, axs)
               for name, (_, axs) in zip(module_names, module_data)]
module_hashes = [hashlib.md5(z).hexdigest() for z in module_zips]

factory_open_file = find_std_file(indir, PORT_NUMS[0], "OPEN")
caldate = gmtime(factory_open_file.stat().st_ctime)
info = {k: v for k, v in (line.split(": ", 1)
                          for line in open(indir / "info.txt", "r").read().split("\n")
                          if ": " in line)}

# --- info.dat header ---
# Binary layout (first 128 bytes), copied from a genuine SEM5032A dump:
#   [ 0:30]  vendor / manufacturer name (NUL-padded)
#   [30:46]  module family (NUL-padded)
#   [46:62]  model / product (NUL-padded)
#   [62:78]  serial number (NUL-padded)
#   [78]     number of ports
#   [79]     0
#   [80:128] 48 zero bytes
# Then an ASCII section beginning with '\n' and containing Connector:,
# Module:, Freq:, Data: and Date: lines.  The record is padded to 1024 bytes
# with '#' characters (0x23), matching the genuine dump.
VENDOR = "Siglent Technologies"
MODEL = "SEM5032A" if NPORTS == 2 else "SEM5004A"
PRODUCT = MODEL
SERIAL = info["Serial"]

header = struct.pack(
    "30s16s16s16sBB48s",
    VENDOR.encode(), MODEL.encode(), PRODUCT.encode(), SERIAL.encode(),
    NPORTS, 0, b"",
)

connector_field = " ".join(["SMA"] * NPORTS)
text = f"\nConnector:{connector_field}"
# Emit one Module/Freq/Data/Date block per requested module.  The first
# module is labelled "Factory" (matching a genuine eCal) and subsequent
# ones are "Factory2", "Factory3", ...  The first field of each Data: line
# is the index N, which the firmware uses to open siglent/data<N>.zip.
# The CSV inside each zip is named after that module, so the SNA's
# name/size/hash validation has a distinct answer per module.  The Freq:
# line reflects the module's own frequency range, since a second module
# may have been taken at higher resolution / different span.
for n, (name, zblob, zhash) in enumerate(
        zip(module_names, module_zips, module_hashes)):
    freqs = module_data[n][0]
    text += (
        f"\nModule:{name}"
        f"\nFreq:{int(freqs[0])},{int(freqs[-1])},{len(freqs)}"
        f"\nData:{n},{len(zblob)},{zhash}"
        f"\nDate:{strftime('%d/%b/%Y', caldate)}"
    )
text += "\n"
header += text.encode()

# Pad to 1024 bytes with '#' to match the genuine device image.  The last
# byte is a '\n', also matching the genuine dump.
header += b"#" * (1024 - len(header) - 1) + b"\n"

(outdir / "siglent").mkdir(exist_ok=True)

for n, zblob in enumerate(module_zips):
    zipname = outdir / f"siglent/data{n}.zip"
    print(f"writing {zipname} ({len(zblob)} bytes, module='{module_names[n]}')",
          file=sys.stderr)
    with open(zipname, "wb") as f:
        f.write(zblob)

datname = outdir / "siglent/info.dat"
print(f"writing {datname}", file=sys.stderr)
with open(datname, "wb") as f:
    f.write(header)
