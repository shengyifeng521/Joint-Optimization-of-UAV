#!/usr/bin/env python3
"""Serial undirected Branch-and-Cut runner with real-time logs and result checks.

Examples:
  python run_experiment.py --data-dir data --exact --no-time-limit --threads 8 --mem-limit-gb 96
  python run_experiment.py --data data/1778.json --exact --no-time-limit
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from check_joint_results import read_json, check_file

ROOT = Path(__file__).resolve().parent
VARIANTS = ("undirected_lazy", "undirected_bc")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group()
    source.add_argument('--data',type=Path)
    source.add_argument('--data-dir',type=Path,default=Path('data'))
    p.add_argument('--out-dir',type=Path,default=Path('result'))
    p.add_argument('--variants',nargs='+',choices=VARIANTS,default=['undirected_bc'])
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--mem-limit-gb',type=float,default=40)
    p.add_argument('--exact',action='store_true')
    p.add_argument('--no-time-limit',action='store_true')
    p.add_argument('--time-limit',type=int,default=3600)
    p.add_argument('--step1-time-limit',type=int,default=1800)
    p.add_argument('--step2-time-limit',type=int,default=1800)
    p.add_argument('--no-two-step-start',action='store_true')
    p.add_argument('--overwrite',action='store_true',help='Explicitly replace existing experiment outputs')
    p.add_argument('--fractional-initial-rounds',type=int,default=20)
    p.add_argument('--fractional-interval',type=int,default=20)
    p.add_argument('--fractional-max-flows',type=int,default=32)
    p.add_argument('--fractional-max-cuts',type=int,default=16)
    p.add_argument('--fractional-budget-seconds',type=float,default=.25)
    args=p.parse_args(argv)
    if args.threads<1 or not math.isfinite(args.mem_limit_gb) or args.mem_limit_gb<=0:
        p.error('Threads and memory must be positive')
    files=[args.data.resolve()] if args.data else sorted(args.data_dir.resolve().glob('*.json'))
    inputs=[]
    for path in files:
        data,sha=read_json(path)
        if not isinstance(data,dict) or not (data.get('view_points') or data.get('viewpoints')):
            print(f'[SKIP] Not an instance: {path.name}',flush=True)
            continue
        inputs.append((path,sha))
        del data
    if not inputs:p.error('No input instances found')
    variants=list(dict.fromkeys(args.variants))
    # Preflight every output before starting an expensive solver.
    for path,sha in inputs:
        for variant in variants:
            result=args.out_dir.resolve()/variant/f'joint_{path.stem}.json'
            record=result.parent/f'{path.stem}.run.json'
            if not args.overwrite and (result.exists() or record.exists()):
                p.error(f'Existing output: {result}; choose a new --out-dir or use --overwrite')
    failed=False
    for path,sha in inputs:
        per_instance=[]
        for variant in variants:
            folder=args.out_dir.resolve()/variant
            folder.mkdir(parents=True,exist_ok=True)
            result=folder/f'joint_{path.stem}.json'
            logfile=folder/f'{path.stem}.log'
            recordfile=folder/f'{path.stem}.run.json'
            lock=folder/f'.{path.stem}.lock'
            command=[sys.executable,'-u',str(ROOT/'joint_copt_undirected.py'),str(path),
                     '--output',str(result),'--threads',str(args.threads),
                     '--mem-limit-gb',str(args.mem_limit_gb),
                     '--time-limit',str(args.time_limit),
                     '--step1-time-limit',str(args.step1_time_limit),
                     '--step2-time-limit',str(args.step2_time_limit)]
            if variant=='undirected_lazy':command+=['--no-fractional-cuts']
            if variant=='undirected_bc':
                for option in ['fractional_initial_rounds','fractional_interval','fractional_max_flows',
                               'fractional_max_cuts','fractional_budget_seconds']:
                    command += ['--'+option.replace('_','-'),str(getattr(args,option))]
            for option in ['exact','no_time_limit','no_two_step_start']:
                if getattr(args,option):command+=['--'+option.replace('_','-')]
            started=time.perf_counter()
            record={'variant':variant,'input':str(path),'input_sha256':sha,'command':command,
                    'solver_source_sha256':digest(command[2]),'result':str(result),'log':str(logfile)}
            # Exclusive creation protects against two tmux workers targeting the same instance.
            with lock.open('x') as stream:stream.write(str(path))
            try:
                if args.overwrite and result.exists():result.unlink()
                print(f'\n[START] {variant}: {path.name}',flush=True)
                with logfile.open('w',encoding='utf-8') as log:
                    log.write(json.dumps(record,ensure_ascii=False)+'\n');log.flush()
                    with subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,text=True,bufsize=1) as process:
                        for line in process.stdout:
                            sys.stdout.write(line);sys.stdout.flush();log.write(line);log.flush()
                        code=process.wait()
                record['returncode']=code
                record['experiment_wall_seconds']=time.perf_counter()-started
                if code==0 and result.exists():
                    saved,result_sha=read_json(result)
                    checked=check_file(path,result)
                    checkfile=folder/f'{path.stem}.check.json'
                    checkfile.write_text(json.dumps(checked,ensure_ascii=False,indent=2))
                    record['constraint_check']=checked['passed'] and checked['data_sha256']==sha
                    record['input_unchanged']=checked['data_sha256']==sha
                    record.update({key:saved.get(key) for key in ['obj','obj_bound','mip_gap','status',
                        'runtime','runtime_two_step','runtime_joint','joint_solver_runtime',
                        'energy_norm_denom','relative_gap_limit','absolute_gap_limit']})
                    failed |= not record['constraint_check']
                else:
                    failed=True;record['constraint_check']=False
                per_instance.append(record)
                print(f'[FINISH] {variant}: {path.name}; returncode={code}; '
                      f'constraints={record.get("constraint_check")}',flush=True)
            finally:
                recordfile.write_text(json.dumps(record,ensure_ascii=False,indent=2))
                lock.unlink(missing_ok=True)
        comparison={'input':str(path),'input_sha256':sha,'runs':per_instance,
                    'note':'Matching objectives certify equivalence empirically only when runs prove optimality.'}
        optimal=[r for r in per_instance if r.get('constraint_check') and r.get('status')==1
                 and r.get('obj') is not None and r.get('obj_bound') is not None
                 and math.isclose(r['obj'],r['obj_bound'],rel_tol=1e-8,abs_tol=1e-8)]
        comparison['compared_zero_gap_variants']=[r['variant'] for r in optimal]
        comparison['all_requested_runs_proven_optimal']=len(optimal)==len(variants)
        if len(optimal)>=2:
            comparison['zero_gap_objectives_match']=all(math.isclose(r['obj'],optimal[0]['obj'],
                rel_tol=1e-8,abs_tol=1e-8) for r in optimal)
            failed |= not comparison['zero_gap_objectives_match']
        comparisonfile=args.out_dir.resolve()/f'{path.stem}.comparison.json'
        comparisonfile.write_text(json.dumps(comparison,ensure_ascii=False,indent=2))
    return int(failed)


if __name__=='__main__':raise SystemExit(main())
