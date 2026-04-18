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
args = parser.parse_args()

indir = args.indir
outdir = args.outdir
NPORTS = args.ports

PORT_NUMS = list(range(1, NPORTS + 1))
PORT_LETTERS = [chr(ord('A') + i) for i in range(NPORTS)]
PAIRS = list(combinations(range(NPORTS), 2))  # index pairs into PORT_NUMS/PORT_LETTERS


def say_open(name, *a):
    print(f"reading {name}", file=sys.stderr)
    return open(name, *a)


def read_s1p(path):
    freqs, re, im = [], [], []
    with say_open(path, "r") as f:
        for l in f:
            l = l.strip()
            if not l or l.startswith("!") or l.startswith("#"):
                continue
            freq, r, i = [float(x) for x in l.split()]
            freqs.append(freq * 1e9)
            re.append(r)
            im.append(i)
    return freqs, re, im


def read_s2p(path):
    freqs = []
    cols = [[] for _ in range(8)]  # s11r,s11i,s21r,s21i,s12r,s12i,s22r,s22i
    with say_open(path, "r") as f:
        for l in f:
            l = l.strip()
            if not l or l.startswith("!") or l.startswith("#"):
                continue
            parts = [float(x) for x in l.split()]
            freqs.append(parts[0] * 1e9)
            for k in range(8):
                cols[k].append(parts[1 + k])
    return freqs, cols


# Read all LibreCAL s1p/s2p data once; the numeric payload is identical
# across modules, only the CSV filename inside the zip changes.
freqs = []
axes = [freqs]

# Per-port one-port standards.
# LibreCAL does not have a physical ATT standard; the firmware does not
# switch state when asked for per-port ATT, so the physical state during
# an ATT request is whatever was last selected (typically LOAD).  Storing
# LOAD values in the ATT slot therefore keeps the stored and measured
# values consistent.
for port in PORT_NUMS:
    load_re, load_im = None, None
    for std in ("OPEN", "SHORT", "LOAD"):
        fr, re_, im_ = read_s1p(indir / f"P{port}_{std}.s1p")
        if not freqs:
            freqs[:] = fr
        axes.append(re_)
        axes.append(im_)
        if std == "LOAD":
            load_re, load_im = re_, im_
    # ATT slot = LOAD data (LibreCAL has no real attenuator).
    axes.append(list(load_re))
    axes.append(list(load_im))

# THROUGH s-parameters for every port pair.
through_cache = {}
for i, j in PAIRS:
    pnum = f"{PORT_NUMS[i]}{PORT_NUMS[j]}"
    fr, cols = read_s2p(indir / f"P{pnum}_THROUGH.s2p")
    through_cache[(i, j)] = cols
    for a in cols:
        axes.append(a)

# CF (confidence-check) s-parameters.  LibreCAL has no separate
# confidence-check standard.  The firmware redirects "SL ATT,n,m" to the
# THRU switch state, so the physical hardware presents the THROUGH
# response in the CF state.  We therefore store the THROUGH data again
# in the CF slot to keep stored and measured values consistent.
for i, j in PAIRS:
    for a in through_cache[(i, j)]:
        axes.append(a)

axes[0] = freqs

# Some early LibreCALs have a truncated P34_THROUGH.s2p file.  The
# Siglent format requires the same number of points for every column,
# so truncate all columns to the shortest one.
shortest = min(len(a) for a in axes)
axes = [a[:shortest] for a in axes]

ar = np.vstack(axes, dtype=np.float64).transpose()

# Build the CSV header row once (depends on NPORTS, not the module).
pair_labels = [PORT_LETTERS[i] + PORT_LETTERS[j] for (i, j) in PAIRS]
csv_header_cols = (["#freq"] + PORT_LETTERS
                   + [f"T_{p}" for p in pair_labels]
                   + [f"CF_{p}" for p in pair_labels]
                   + ["END"])

info_txt_lines = open(indir / "info.txt", "r").read().splitlines()


def build_module_zip(module_name):
    """Produce a zip blob whose single entry is '<module_name>.csv' with
    the full LibreCAL Siglent payload inside."""
    buf = io.BytesIO()
    with ZipFile(buf, "w", compression=ZIP_DEFLATED, compresslevel=9) as zf, \
         zf.open(f"{module_name}.csv", "w") as outf_b, \
         io.TextIOWrapper(outf_b, newline='\x0A') as outf:
        for l in info_txt_lines:
            outf.write(f"! {l.strip()}\n")
        outf.write(",".join(csv_header_cols) + "\n")
        np.savetxt(outf, ar, delimiter=",")
    return buf.getvalue()


module_names = ["Factory" if n == 0 else f"Factory{n + 1}"
                for n in range(args.modules)]
module_zips = [build_module_zip(name) for name in module_names]
module_hashes = [hashlib.md5(z).hexdigest() for z in module_zips]
print(f"compressing {ar.shape[0]} points with {ar.shape[1]} columns "
      f"into {args.modules} module zip(s)")

caldate = gmtime((indir / "P1_OPEN.s1p").stat().st_ctime)
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
# name/size/hash validation has a distinct answer per module.
for n, (name, zblob, zhash) in enumerate(zip(module_names, module_zips, module_hashes)):
    text += (
        f"\nModule:{name}"
        f"\nFreq:{int(axes[0][0])},{int(axes[0][-1])},{len(axes[0])}"
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
