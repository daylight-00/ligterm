"""Tests of 00_fetch_data.py on a small synthetic package: the flat layout of a Zenodo record, the default selection (datasets and results; only the selected archives are verified), the manifest checked against
DATA_PACKAGE.json, a damaged archive, an unknown name, a member that leaves the data folder, and --check. It needs the `zstd` command; without it the tests are skipped (exit code 0).
usage: python3 -B test_fetch_data.py        (standard library only; prints 'n/N checks passed'; exit 1 on any failure)"""
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "scripts", "00_fetch_data.py")
N_OK = N_ALL = 0


def ok(cond, msg):
    global N_OK, N_ALL
    N_ALL += 1
    if cond:
        N_OK += 1
    else:
        print("FAIL:", msg)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def make_archive(path, members):
    """members: {name in the archive: bytes}; a tar.zst written by tarfile and zstd"""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in members.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    z = subprocess.run(["zstd", "-q", "-c"], input=buf.getvalue(), stdout=subprocess.PIPE, check=True)
    open(path, "wb").write(z.stdout)


def make_package(src, evil=False):
    """a flat record folder src and its DATA_PACKAGE.json (returned as a path)"""
    os.makedirs(src)
    files = {"datasets/a.txt": b"alpha\n", "results/b.txt": b"beta\n", "intermediates/x/c.txt": b"gamma\n", "intermediates/y/d.txt": b"delta\n"}
    groups = {"datasets.tar.zst": ["datasets/a.txt"], "results.tar.zst": ["results/b.txt"], "x.tar.zst": ["intermediates/x/c.txt"], "y.tar.zst": ["intermediates/y/d.txt"]}
    for name, parts in groups.items():
        mem = {p: files[p] for p in parts}
        if evil and name == "results.tar.zst":
            mem["../escape.txt"] = b"outside\n"
        make_archive(os.path.join(src, name), mem)
    man = "".join(f"{sha(files[p])}  {p}\n" for p in sorted(files))
    open(os.path.join(src, "MANIFEST.sha256"), "w").write(man)
    pkg = dict(record_files_url=None, manifest_sha256=sha(man.encode()), intermediates=["x", "y"],
               files={n: dict(bytes=os.path.getsize(os.path.join(src, n)), sha256=sha(open(os.path.join(src, n), "rb").read())) for n in groups})
    pj = src + ".json"
    json.dump(pkg, open(pj, "w"))
    return pj


def run(data, src, pj, *extra):
    r = subprocess.run([sys.executable, "-B", SCRIPT, "--data-dir", data, "--source", src, "--package", pj, *extra], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


if shutil.which("zstd") is None:
    print("skipped: the `zstd` command is not installed")
    sys.exit(0)

with tempfile.TemporaryDirectory() as tmp:
    src = os.path.join(tmp, "record")
    pj = make_package(src)
    # 1. the default: datasets and results only, and nothing else is demanded of the manifest
    d1 = os.path.join(tmp, "d1")
    rc, out = run(d1, src, pj)
    ok(rc == 0 and os.path.exists(f"{d1}/datasets/a.txt") and os.path.exists(f"{d1}/results/b.txt") and not os.path.exists(f"{d1}/intermediates"), f"default fetch: {rc} {out[-200:]}")
    ok("2 files of datasets, results verified" in out, f"default fetch verifies only the selected archives: {out[-200:]}")
    ok(not os.path.exists(f"{d1}/_download"), "the downloaded archives are removed")
    # 2. one intermediate, then all
    d2 = os.path.join(tmp, "d2")
    rc, out = run(d2, src, pj, "--intermediates", "x")
    ok(rc == 0 and os.path.exists(f"{d2}/intermediates/x/c.txt") and not os.path.exists(f"{d2}/intermediates/y") and "3 files" in out, f"one intermediate: {rc} {out[-200:]}")
    d3 = os.path.join(tmp, "d3")
    rc, out = run(d3, src, pj, "--intermediates", "all")
    ok(rc == 0 and os.path.exists(f"{d3}/intermediates/y/d.txt") and "4 files" in out, f"all intermediates: {rc} {out[-200:]}")
    # 3. --check: what is unpacked is verified, a changed file and a changed manifest are found
    rc, out = run(d1, src, pj, "--check")
    ok(rc == 0 and "2 files" in out, f"check of the default data: {rc} {out[-200:]}")
    rc, out = run(d3, src, pj, "--check")
    ok(rc == 0 and "4 files" in out, f"check of all data: {rc} {out[-200:]}")
    f = f"{d3}/datasets/a.txt"
    keep = open(f, "rb").read()
    open(f, "ab").write(b"x")
    rc, out = run(d3, src, pj, "--check")
    ok(rc == 1 and "datasets/a.txt differs" in out, f"a changed file is found: {rc} {out[-200:]}")
    open(f, "wb").write(keep)
    m = f"{d3}/MANIFEST.sha256"
    keep_m = open(m).read()
    open(m, "a").write(f"{sha(b'z')}  results/none.txt\n")
    rc, out = run(d3, src, pj, "--check")
    ok(rc == 1 and "MANIFEST.sha256 differs" in out, f"a changed manifest is found: {rc} {out[-200:]}")
    open(m, "w").write(keep_m)
    # 4. a tampered manifest in the source is refused before anything is unpacked
    src2 = os.path.join(tmp, "record_tampered")
    shutil.copytree(src, src2)
    open(os.path.join(src2, "MANIFEST.sha256"), "a").write(f"{sha(b'q')}  results/extra.txt\n")
    d4 = os.path.join(tmp, "d4")
    rc, out = run(d4, src2, pj)
    ok(rc != 0 and "MANIFEST.sha256 differs" in out and not os.path.exists(f"{d4}/datasets") and not os.path.exists(f"{d4}/MANIFEST.sha256"), f"tampered manifest: {rc} {out[-200:]}")
    # 5. a damaged archive is refused and deleted
    src3 = os.path.join(tmp, "record_damaged")
    shutil.copytree(src, src3)
    p = os.path.join(src3, "x.tar.zst")
    b = bytearray(open(p, "rb").read())
    b[len(b) // 2] ^= 1
    open(p, "wb").write(bytes(b))
    d5 = os.path.join(tmp, "d5")
    rc, out = run(d5, src3, pj, "--intermediates", "x")
    ok(rc != 0 and "x.tar.zst" in out and "SHA-256" in out and not os.path.exists(f"{d5}/_download/x.tar.zst"), f"damaged archive: {rc} {out[-200:]}")
    # 6. an unknown name
    rc, out = run(os.path.join(tmp, "d6"), src, pj, "--intermediates", "z")
    ok(rc != 0 and "not in the package" in out, f"unknown intermediate: {rc} {out[-200:]}")
    # 7. a member that leaves the data folder
    src7 = os.path.join(tmp, "record_evil")
    pj7 = make_package(src7, evil=True)
    d7 = os.path.join(tmp, "d7", "data")
    rc, out = run(d7, src7, pj7)
    ok(rc != 0 and "refusing the member" in out and not os.path.exists(os.path.join(tmp, "d7", "escape.txt")), f"member that leaves the folder: {rc} {out[-200:]}")
    # 8. the layout of a record: files only, flat names
    ok(all(os.path.isfile(os.path.join(src, n)) for n in os.listdir(src)) and all("/" not in n for n in json.load(open(pj))["files"]), "the synthetic record is flat")

# 9. the real description of the package: flat names, and the intermediates it lists are the archives it holds
real = json.load(open(os.path.join(HERE, "..", "DATA_PACKAGE.json")))
ok(all("/" not in n for n in real["files"]), "DATA_PACKAGE.json names no sub-folder")
ok(set(real["files"]) == {"datasets.tar.zst", "results.tar.zst"} | {f"{n}.tar.zst" for n in real["intermediates"]}, "DATA_PACKAGE.json lists the archives of its intermediates")
ok(len(real["manifest_sha256"]) == 64, "DATA_PACKAGE.json carries the SHA-256 of the manifest")

print(f"{N_OK}/{N_ALL} checks passed")
sys.exit(0 if N_OK == N_ALL else 1)
