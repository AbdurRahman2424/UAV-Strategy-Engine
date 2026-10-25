"""Portable JSON/CSV output; no optional dataframe dependency."""
import csv
import json
from pathlib import Path
import numpy as np


def json_default(obj):
    if isinstance(obj,np.generic):
        return obj.item()
    if isinstance(obj,np.ndarray):
        return obj.tolist()
    raise TypeError(f'Cannot serialize {type(obj)}')


def write_json(path,data):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,default=json_default)+'\n',encoding='utf-8')


def write_csv(path,rows):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        raise ValueError('No rows to write.')
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
