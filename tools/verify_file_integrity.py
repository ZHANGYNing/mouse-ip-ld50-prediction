"""Verify the distributed file manifest after download or upload."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]


def main():
    manifest=json.loads((ROOT/'provenance/package_files.json').read_text())
    errors=[]
    for item in manifest['files']:
        path=ROOT/item['path']
        if not path.is_file():errors.append('Missing: '+item['path']);continue
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=item['sha256']:errors.append('Changed: '+item['path'])
    if errors:raise SystemExit('\n'.join(errors))
    print(f"PASS: {len(manifest['files'])} packaged files match their recorded SHA256 hashes.")


if __name__=='__main__':main()
