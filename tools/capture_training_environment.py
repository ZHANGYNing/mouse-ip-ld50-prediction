"""Run inside the original training environment to record relevant package versions."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import importlib.metadata
import json
import platform
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'environment/captured_training_environment.json')
    args=p.parse_args()
    versions={}
    for name in ['numpy','pandas','scipy','scikit-learn','rdkit','xgboost','xgboost-cpu','joblib','tqdm','openpyxl','matplotlib']:
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]=None
    report={'captured_at_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,
            'platform':platform.platform(),'packages':versions,
            'scope':'Versions in the environment where this utility was run. The author must establish whether this is the unchanged original training environment.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('Environment written to',args.output.resolve())


if __name__=='__main__':main()
