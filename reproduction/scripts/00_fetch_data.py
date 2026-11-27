#!/usr/bin/env python3
"""Fetch the data of the reproduction: the research data package (Zenodo) and, for three of the experiments, the third-party NetCleave files.

Research data package: a Zenodo record holds only top-level files. `datasets.tar.zst` (the units, sequences, sequence groups and the study profiles; the external cohort tables) and `results.tar.zst` (the per-study
result records of every experiment) are the default; every stored input of the later experiments is one more archive `<name>.tar.zst` (an intermediate: name it with --intermediates, or give `all`). Unpacking an archive
gives `datasets/`, `results/` or `intermediates/<name>/` in the data folder. `DATA_PACKAGE.json` (next to `configs/`) lists the archives with their sizes and SHA-256 and the SHA-256 of `MANIFEST.sha256`: the manifest is
verified against it, every archive is verified before it is unpacked, and the unpacked files of the selected archives (only those) are verified against the manifest. --source is the file folder of the Zenodo record
(a URL) or a local folder that holds the files (a mirror, a copy on a cluster file system). The record is in preparation: until `DATA_PACKAGE.json` carries its address, give --source.

NetCleave (third party, not redistributed): scripts 20, 21 and 22 compare with the NetCleave predictor of the BSC fork; they read the descriptor table, the class-II weights and the training sequences of
the public repository, pinned here to one commit and to the SHA-256 of every file (`configs/third_party.json`). The weights are stored there as a Keras .h5 file; they are converted to the .npz
arrays that the scripts read (h5py is needed for this one step; the arrays are checked against a hash). Pass the folder as --netcleave-dir of scripts 20, 21 and 22.

usage: 00_fetch_data.py --data-dir <folder> [--source URL_OR_FOLDER] [--intermediates NAME ... | all] [--check]
       00_fetch_data.py --netcleave-dir <folder>
Needs the `zstd` command (version 1.4 or newer) for the data package; Python 3.10 or newer, standard library only (h5py and numpy for the NetCleave conversion).
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request

here = os.path.dirname(os.path.abspath(__file__))
CFG = os.path.join(here, "..", "configs")
PACKAGE = os.path.join(here, "..", "DATA_PACKAGE.json")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def fetch(src, name, dest):
    """copy or download one top-level file of the record into dest"""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.isdir(src):
        shutil.copyfile(os.path.join(src, name), dest)
        return
    url = src.rstrip("/") + "/" + name + ("?download=1" if "zenodo.org" in src else "")
    with urllib.request.urlopen(url, timeout=300) as r, open(dest, "wb") as out:
        shutil.copyfileobj(r, out, 1 << 20)


def unpack(archive, dest):
    """a tar.zst archive into dest; members that would leave dest are refused"""
    if shutil.which("zstd") is None:
        sys.exit("the `zstd` command is not installed (version 1.4 or newer is needed): apt install zstd, brew install zstd, ...")
    root = os.path.realpath(dest)
    p = subprocess.Popen(["zstd", "-d", "-q", "-c", archive], stdout=subprocess.PIPE)
    with tarfile.open(fileobj=p.stdout, mode="r|") as tf:
        for m in tf:
            target = os.path.realpath(os.path.join(dest, m.name))
            if not (target == root or target.startswith(root + os.sep)) or not (m.isfile() or m.isdir()):
                sys.exit(f"{archive}: refusing the member {m.name!r}")
            tf.extract(m, dest, filter="data")
    if p.wait() != 0:
        sys.exit(f"zstd could not decompress {archive}")


def group_of(rel):
    """the archive that a file of the manifest belongs to: datasets, results or intermediates/<name>"""
    parts = rel.split("/")
    return "/".join(parts[:2]) if parts[0] == "intermediates" and len(parts) > 2 else parts[0]


def check_manifest(data_dir, groups=None):
    """the files of MANIFEST.sha256 that belong to `groups` (default: the groups that are unpacked in data_dir) must exist with the listed hash; returns (n checked, problems, groups checked)"""
    man = os.path.join(data_dir, "MANIFEST.sha256")
    if not os.path.exists(man):
        return 0, ["MANIFEST.sha256 is not in the data folder"], []
    entries = []
    for ln in open(man):
        if ln.strip():
            h, rel = ln.rstrip("\n").split("  ", 1)
            entries.append((h, rel))
    if groups is None:
        groups = sorted({group_of(rel) for _h, rel in entries if os.path.exists(os.path.join(data_dir, group_of(rel)))})
    n, bad = 0, []
    for h, rel in entries:
        if group_of(rel) not in groups:
            continue
        p = os.path.join(data_dir, rel)
        if not os.path.exists(p):
            bad.append(f"{rel} is missing")
        else:
            n += 1
            if sha256_file(p) != h:
                bad.append(f"{rel} differs from the manifest")
    return n, bad, list(groups)


def verify_manifest_file(path, pkg):
    if sha256_file(path) != pkg["manifest_sha256"]:
        return f"MANIFEST.sha256 differs from the SHA-256 in DATA_PACKAGE.json (a damaged download or another version of the package)"
    return None


def fetch_data(a):
    pkg = json.load(open(a.package))
    src = a.source or pkg.get("record_files_url")
    if not src:
        sys.exit("the data record (Zenodo) is in preparation: DATA_PACKAGE.json has no address for it yet; give --source (a folder with the files of the record or the URL of its file folder)")
    known = pkg["intermediates"]
    chosen = known if a.intermediates == ["all"] else (a.intermediates or [])
    unknown = [n for n in chosen if n not in known]
    if unknown:
        sys.exit(f"not in the package: {unknown}; available: {known}")
    want = ["datasets.tar.zst", "results.tar.zst"] + [f"{n}.tar.zst" for n in chosen]
    os.makedirs(a.data_dir, exist_ok=True)
    manifest = os.path.join(a.data_dir, "MANIFEST.sha256")
    fetch(src, "MANIFEST.sha256", manifest)
    why = verify_manifest_file(manifest, pkg)
    if why:
        os.remove(manifest)
        sys.exit(why + "; the file was deleted")
    for name in want:
        dest = os.path.join(a.data_dir, "_download", name)
        if not (os.path.exists(dest) and sha256_file(dest) == pkg["files"][name]["sha256"]):
            print(f"fetching {name} ({pkg['files'][name]['bytes'] / 1e6:.1f} MB)", flush=True)
            fetch(src, name, dest)
        if sha256_file(dest) != pkg["files"][name]["sha256"]:
            os.remove(dest)
            sys.exit(f"{name}: the SHA-256 differs from DATA_PACKAGE.json (a damaged download or another version of the file); the file was deleted")
        print(f"unpacking {name}", flush=True)
        unpack(dest, a.data_dir)
    groups = ["datasets", "results"] + [f"intermediates/{n}" for n in chosen]
    n, bad, _ = check_manifest(a.data_dir, groups=groups)
    print(f"{n} files of {', '.join(groups)} verified against MANIFEST.sha256" + (f"; {len(bad)} problems" if bad else ""))
    for b in bad[:10]:
        print("  ", b)
    shutil.rmtree(os.path.join(a.data_dir, "_download"), ignore_errors=True)
    return 1 if bad else 0


def check_data(a):
    pkg = json.load(open(a.package))
    manifest = os.path.join(a.data_dir, "MANIFEST.sha256")
    if os.path.exists(manifest):
        why = verify_manifest_file(manifest, pkg)
        if why:
            print(why)
            return 1
    n, bad, groups = check_manifest(a.data_dir)
    print(f"{n} files of {', '.join(groups) or 'no unpacked archive'} verified" + (f"; {len(bad)} problems: {bad[:5]}" if bad else ""))
    return 1 if (bad or not groups) else 0


def array_hash(npz_path):
    import numpy as np
    z = np.load(npz_path)
    h = hashlib.sha256()
    for k in sorted(z.files):
        arr = np.ascontiguousarray(z[k])
        h.update(k.encode() + b"\0" + str(arr.dtype).encode() + b"\0" + str(arr.shape).encode() + b"\0" + arr.tobytes())
    return h.hexdigest()


def fetch_netcleave(a):
    cfg = json.load(open(os.path.join(CFG, "third_party.json")))["netcleave"]
    base = f"{cfg['raw_url']}/{cfg['commit']}/"
    for f in cfg["files"]:
        dest = os.path.join(a.netcleave_dir, f["save_as"])
        if os.path.exists(dest) and sha256_file(dest) == f["sha256"]:
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print(f"fetching {f['path']}", flush=True)
        with urllib.request.urlopen(base + f["path"], timeout=300) as r, open(dest + ".part", "wb") as out:
            shutil.copyfileobj(r, out, 1 << 20)
        if sha256_file(dest + ".part") != f["sha256"]:
            os.remove(dest + ".part")
            sys.exit(f"{f['path']}: the SHA-256 differs from the one recorded for commit {cfg['commit'][:10]}")
        os.replace(dest + ".part", dest)
    w = cfg["weights_npz"]
    npz = os.path.join(a.netcleave_dir, w["to"])
    if not (os.path.exists(npz) and array_hash(npz) == w["arrays_sha256"]):
        try:
            import h5py
            import numpy as np
        except ImportError:
            sys.exit("h5py is needed once, to convert the weights of NetCleave to numpy arrays: pip install h5py")
        out = {}
        with h5py.File(os.path.join(a.netcleave_dir, w["from"]), "r") as fh:
            names = []
            fh.visititems(lambda n, o: names.append(n) if isinstance(o, h5py.Dataset) else None)
            for n in names:
                out[n.replace("/", "__")] = np.array(fh[n])
        np.savez(npz, **out)
        if array_hash(npz) != w["arrays_sha256"]:
            os.remove(npz)
            sys.exit("the converted weights differ from the arrays that the experiments were run with (another h5py or numpy version?)")
    print(f"NetCleave files in {a.netcleave_dir} (commit {cfg['commit'][:10]}): descriptor table, weights as arrays, training sequences; all verified")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir")
    ap.add_argument("--source")
    ap.add_argument("--intermediates", nargs="*", help="names of the intermediates to fetch, or `all`")
    ap.add_argument("--check", action="store_true", help="only verify the unpacked data against MANIFEST.sha256 (and the manifest against DATA_PACKAGE.json)")
    ap.add_argument("--package", default=PACKAGE, help="the description of the data package (default: DATA_PACKAGE.json)")
    ap.add_argument("--netcleave-dir")
    a = ap.parse_args()
    if not a.data_dir and not a.netcleave_dir:
        ap.error("give --data-dir and/or --netcleave-dir")
    code = 0
    if a.data_dir:
        code |= check_data(a) if a.check else fetch_data(a)
    if a.netcleave_dir:
        code |= fetch_netcleave(a)
    return code


if __name__ == "__main__":
    sys.exit(main())
