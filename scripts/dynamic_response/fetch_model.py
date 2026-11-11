"""Install the SHA-256-verified public prediction bundle into the checkout."""
import hashlib
import shutil
import tarfile
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

from project_paths import configured_path

ASSET = 'https://github.com/dong-zhongyuan/ad-world-model/releases/download/v3.0.1/dynamic_model_v3.tar.gz'
SHA256 = 'c426f665b4e0b93f77810b2f8f685d95a59f078651cfe31649ab542dcfc01e0f'
PACKAGE_PATH = 'checkpoints/dynamic_response_v3/'
FILES = ('backbone.pt', 'heads.npz', 'metadata.json', 'dynamic_model.py',
         'model_code/dynamics.py', 'model_code/unified_model_v3.py', 'model_code/world_model.py')


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as reader:
        for block in iter(lambda: reader.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    target = configured_path('model_package')
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        archive = Path(tmp) / 'bundle.tgz'
        req = Request(ASSET, headers={'User-Agent': 'ad-world-model-benchmark'})
        print('Downloading the frozen model (115 MB)...', flush=True)
        with urlopen(req, timeout=120) as src, archive.open('wb') as dst:
            shutil.copyfileobj(src, dst)
        if checksum(archive) != SHA256:
            raise ValueError('Release archive checksum mismatch')
        package = Path(tmp) / 'model'
        with tarfile.open(archive, 'r:gz') as bundle:
            for name in FILES:
                member = bundle.getmember(PACKAGE_PATH + name)
                if not member.isfile():
                    raise ValueError(f'Expected regular file: {name}')
                outfile = package / name
                outfile.parent.mkdir(exist_ok=True, parents=True)
                with bundle.extractfile(member) as src, outfile.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
        if target.exists():
            for name in FILES:
                if not (target / name).is_file() or checksum(target / name) != checksum(package / name):
                    raise ValueError(f'Existing installation differs at {target / name}')
        else:
            package.rename(target)
    print(f'Frozen benchmark model verified at {target}')


if __name__ == '__main__':
    main()
