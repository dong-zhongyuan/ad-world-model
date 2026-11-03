import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import threading
from pathlib import Path

print_lock = threading.Lock()


def remote_metadata(url):
    result = subprocess.run(
        ["curl", "-fIL", "--connect-timeout", "30", "-sS", url],
        check=True,
        capture_output=True,
        text=True,
    )
    size = None
    etag = None
    for line in result.stdout.splitlines():
        if line.lower().startswith("content-length:"):
            size = int(line.split(":", 1)[1].strip())
        if line.lower().startswith("etag:"):
            etag = line.split(":", 1)[1].strip().strip('"')
    if size is None:
        raise RuntimeError("Remote Content-Length is unavailable")
    return size, etag


def download_part(url, directory, index, start, end):
    path = directory / f"part_{index:05d}"
    expected = end - start + 1
    if path.exists() and path.stat().st_size == expected:
        with print_lock:
            print(f"SKIP {index} {expected}", flush=True)
        return path
    temporary = path.with_suffix(".tmp")
    command = [
        "curl", "-fL", "--retry", "8", "--connect-timeout", "30",
        "--speed-time", "60", "--speed-limit", "1024", "-sS",
        "--range", f"{start}-{end}", url, "-o", str(temporary),
    ]
    subprocess.run(command, check=True)
    actual = temporary.stat().st_size
    if actual != expected:
        raise RuntimeError(f"Part {index}: expected {expected} bytes, received {actual}")
    os.replace(temporary, path)
    with print_lock:
        print(f"DONE {index} {expected}", flush=True)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--part-mb", type=int, default=64)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    size, etag = remote_metadata(args.url)
    part_size = args.part_mb * 1024 * 1024
    ranges = [(i, start, min(start + part_size - 1, size - 1)) for i, start in enumerate(range(0, size, part_size))]
    directory = output.parent / (output.name + ".parts")
    directory.mkdir(exist_ok=True)
    metadata = {"url": args.url, "size": size, "etag": etag, "part_size": part_size, "parts": len(ranges)}
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_part, args.url, directory, *item) for item in ranges]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    assembling = output.with_suffix(output.suffix + ".assembling")
    with assembling.open("wb") as target:
        for index, _, _ in ranges:
            with (directory / f"part_{index:05d}").open("rb") as source:
                shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
    if assembling.stat().st_size != size:
        raise RuntimeError("Assembled file size mismatch")
    os.replace(assembling, output)
    (output.parent / (output.name + ".download.json")).write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"COMPLETE {output} {size}", flush=True)


if __name__ == "__main__":
    main()
