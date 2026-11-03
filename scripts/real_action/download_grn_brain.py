import concurrent.futures
import subprocess
import tarfile
from pathlib import Path

WORK = Path(__file__).resolve().parent
DATA = WORK / "data/grn_brain"
ARCHIVES = DATA / "archives"
RAW = DATA / "raw"
BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE272nnn/GSE272377/suppl"


def remote_size(url):
    result = subprocess.run(["curl", "-fIL", "--connect-timeout", "20", "--max-time", "60", "-sS", url], check=True, capture_output=True, text=True)
    sizes = [int(line.split(":", 1)[1].strip()) for line in result.stdout.splitlines() if line.lower().startswith("content-length:")]
    if not sizes:
        raise RuntimeError(f"No Content-Length for {url}")
    return sizes[-1]


def download(url, output):
    expected = remote_size(url)
    if output.exists() and output.stat().st_size == expected:
        print(f"SKIP {output.name}", flush=True)
        return output
    command = ["curl", "-fL", "--retry", "10", "--connect-timeout", "20", "--speed-time", "60", "--speed-limit", "1024", "-sS"]
    if output.exists() and output.stat().st_size < expected:
        command += ["-C", "-"]
    command += [url, "-o", str(output)]
    subprocess.run(command, check=True)
    if output.stat().st_size != expected:
        raise RuntimeError(f"Size mismatch for {output.name}: {output.stat().st_size} versus {expected}")
    print(f"DONE {output.name} {expected}", flush=True)
    return output


def main():
    ARCHIVES.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    names = ["GSE272377_feature_config.csv.gz"] + [f"GSE272377_HSA-0000{identifier}_pool_{pool}.tar.gz" for identifier in range(33307, 33317) for pool in ["A", "C"]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(download, f"{BASE}/{name}", ARCHIVES / name) for name in names]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    for path in sorted(ARCHIVES.glob("*.tar.gz")):
        with tarfile.open(path, "r:gz") as archive:
            for member in archive.getmembers():
                if not (RAW / member.name).resolve().is_relative_to(RAW.resolve()):
                    raise RuntimeError(f"Unsafe archive path: {member.name}")
            archive.extractall(RAW)
    files = sorted(RAW.rglob("*"))
    (DATA / "file_inventory.txt").write_text("\n".join(f"{path} {path.stat().st_size} bytes" for path in files if path.is_file()) + "\n")
    print(f"GRN DATA READY: {sum(path.is_file() for path in files)} files", flush=True)


if __name__ == "__main__":
    main()
