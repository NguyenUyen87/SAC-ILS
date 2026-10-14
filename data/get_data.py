"""Download and unpack the benchmark of Qiu (2025) (Science Data Bank,
https://www.scidb.cn/en/s/ZfauMn): Instances.rar (small and large sets, 2.2 GB) and
Instances-scalability.rar (scalability set, 0.4 GB). Downloads resume after an
interruption and are checked against their MD5. Only the folders used here are
unpacked (about 4.5 GB): Instances/SMTI/n_100_* and n_1000_* to <data>/Instances/SMTI,
and the three scalability folders to <data>/scal/Instances/SMTI.

The archives are RAR 5. Unpacking uses the first tool found: 7-Zip (7z, also
C:\\Program Files\\7-Zip\\7z.exe), unrar, or bsdtar (the tar of Windows 10 and macOS). On Windows,
install 7-Zip if none works (https://www.7-zip.org, or: winget install 7zip.7zip).

    python data/get_data.py                 # into data/
    python data/get_data.py --data D:/smti  # elsewhere (then set SACSMTI_DATA or use run_experiments.py --data)
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "https://download.scidb.cn/download?dataSetType=personal&fileId={id}&fileName={name}"
ARCHIVES = [
    {"name": "Instances.rar", "id": "a6aeb04072858030ecde6c06cdd30396", "size": 2336490366,
     "md5": "11617fe1d872133214007d3f8299e239", "into": ".",
     "members": ["Instances/SMTI/n_100_*", "Instances/SMTI/n_1000_*"]},
    {"name": "Instances-scalability.rar", "id": "d50b5ac8d9d8b368aaef6def1ab34342", "size": 448259877,
     "md5": "57398311235b1f131dea14cbec5e46e3", "into": "scal",
     "members": [f"Instances/SMTI/n_{n}_p1_10_p2_0.5_geometric/*" for n in (2000, 5000, 10000)]},
]
CHUNK = 1 << 20


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def _segment(url, part, start, end, progress):
    """Fetch bytes start..end (inclusive) into the file `part`, resuming and retrying."""
    want = end - start + 1
    for attempt in range(1, 10 ** 6):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have >= want:
            return
        req = urllib.request.Request(url, headers={"Range": f"bytes={start + have}-{end}", "User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.status != 206:
                    raise OSError(f"HTTP {r.status} instead of a partial response")
                with open(part, "ab") as fh:
                    while True:
                        b = r.read(CHUNK)
                        if not b:
                            break
                        fh.write(b)
                        progress[part] = have + fh.tell()
        except OSError as e:
            if attempt % 10 == 1:
                print(f"  {os.path.basename(part)}: {e}; retrying", flush=True)
            time.sleep(min(60, 5 * attempt))


def download(a, path, segments=8):
    """Download in `segments` parallel byte ranges (the server is slow per connection);
    the parts are kept until the whole file is there, so an interrupted run resumes."""
    import threading
    url = BASE.format(**a)
    size = a["size"]
    step = -(-size // segments)
    parts = [(f"{path}.part{i}", i * step, min(size, (i + 1) * step) - 1) for i in range(segments)]
    progress = {p: (os.path.getsize(p) if os.path.exists(p) else 0) for p, _, _ in parts}
    threads = [threading.Thread(target=_segment, args=(url, p, s, e, progress), daemon=True) for p, s, e in parts]
    for t in threads:
        t.start()
    t0, b0 = time.time(), sum(progress.values())
    while any(t.is_alive() for t in threads):
        time.sleep(30)
        got = sum(progress.values())
        rate = (got - b0) / max(time.time() - t0, 1e-6) / 2 ** 20
        print(f"  {a['name']}: {got / 2 ** 30:.2f}/{size / 2 ** 30:.2f} GB, {rate:.2f} MB/s", flush=True)
    with open(path, "wb") as out:
        for p, _, _ in parts:
            with open(p, "rb") as fh:
                shutil.copyfileobj(fh, out, CHUNK)
    for p, _, _ in parts:
        os.remove(p)


def extractor():
    """Command prefix that lists a RAR 5 archive, or None."""
    cands = ["7z", "7za", r"C:\Program Files\7-Zip\7z.exe", r"C:\Program Files (x86)\7-Zip\7z.exe", "unrar", "bsdtar", "tar"]
    for c in cands:
        exe = shutil.which(c) or (c if os.path.isfile(c) else None)
        if exe:
            return os.path.basename(exe).lower().split(".")[0], exe
    return None


def unpack(archive, dest, members):
    tool = extractor()
    if tool is None:
        raise SystemExit("no tool to unpack RAR 5 archives: install 7-Zip (https://www.7-zip.org) and rerun")
    kind, exe = tool
    os.makedirs(dest, exist_ok=True)
    if kind in ("7z", "7za"):
        cmd = [exe, "x", "-y", f"-o{dest}", archive, *[m.replace("/", os.sep) for m in members]]
    elif kind == "unrar":
        cmd = [exe, "x", "-o+", archive, *members, dest + os.sep]
    else:                                    # bsdtar (the tar.exe of Windows 10 and macOS is bsdtar)
        cmd = [exe, "-xf", archive, "-C", dest, *sum((["--include", m] for m in members), [])]
    print(" ".join(cmd), flush=True)
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL)
    if r.returncode != 0:
        raise SystemExit(f"unpacking failed ({kind}); install 7-Zip (https://www.7-zip.org) and rerun")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=HERE, help="data directory (default: data/)")
    ap.add_argument("--keep", action="store_true", help="keep the archives after unpacking")
    args = ap.parse_args()
    data = os.path.abspath(args.data)
    os.makedirs(data, exist_ok=True)
    for a in ARCHIVES:
        dest = os.path.join(data, a["into"])
        check = os.path.join(dest, "Instances", "SMTI")
        marker = os.path.join(dest, f".{a['name']}.ok")
        if os.path.exists(marker):
            print(f"{a['name']}: already unpacked in {check}")
            continue
        path = os.path.join(data, a["name"])
        if not (os.path.exists(path) and a["size"] and os.path.getsize(path) == a["size"] and md5(path) == a["md5"]):
            print(f"downloading {a['name']} ...", flush=True)
            download(a, path)
            got = md5(path)
            if got != a["md5"]:
                os.remove(path)
                raise SystemExit(f"{a['name']}: MD5 {got} != {a['md5']}; the file was removed, rerun to download again")
        print(f"{a['name']}: MD5 ok, unpacking ...", flush=True)
        unpack(path, dest, a["members"])
        folders = sorted(os.listdir(check))
        print(f"  {len(folders)} folders in {check}")
        open(marker, "w").close()
        if not args.keep:
            os.remove(path)
    print("data ready in", data)


if __name__ == "__main__":
    sys.exit(main())
