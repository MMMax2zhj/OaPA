"""Bounded same-GPU orchestration around the unchanged, M0-accepted experiment code.

Workers use separate Python/CUDA processes and retain the original seed/config.
The running serial trainer can be adopted without terminating or restarting it.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from collections import Counter
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
import oapa4r as core
from oapa4r_reporting import install
core.JOBS = [j for j in core.JOBS if j['block'] in ('MAIN_P0','MAIN_P1')]
assert len(core.JOBS) == 72
install(core)
FOLLOW_WORKERS = {}



def now():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())


def proc_args(pid):
    p=Path('/proc')/str(pid)
    try:
        if (p/'stat').read_text().split(') ',1)[1].split()[0]=='Z':return []
        if (p/'cwd').resolve()!=ROOT:return []
        return [x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x]
    except (FileNotFoundError,PermissionError,ProcessLookupError):return []


def output_argument(args):
    return args[args.index('--output')+1] if '--output' in args else None


def trainers():
    result={}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        args=proc_args(int(path.name))
        if 'code/scripts/train_oapa.py' in args:
            result[output_argument(args)]=int(path.name)
    return result


def gpu():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu,power.draw',
                                  '--format=csv,noheader,nounits'],text=True).strip().splitlines()[0]
    used,free,util,power=[float(s.strip()) for s in text.split(',')]
    return {'time_utc':now(),'memory_used_mib':used,'memory_free_mib':free,'utilization_percent':util,'power_w':power}


def setup():
    core.torch.set_num_threads(4)
    core.torch.backends.cudnn.benchmark=False;core.torch.backends.cudnn.deterministic=True
    core.torch.backends.cuda.matmul.allow_tf32=False


def worker(identifier,adopt_pid=None,follow_pid=None):
    setup();job=next(j for j in core.JOBS if j['id']==identifier)
    folder=ROOT/job['run_dir']
    if follow_pid:
        while proc_args(follow_pid):
            args=proc_args(follow_pid)
            if '--worker-run' not in args or args[args.index('--worker-run')+1]!=identifier:
                raise RuntimeError('Existing worker identity changed')
            time.sleep(2)
        record=core.read(folder/'parallel_worker_complete.json')
        if record['run_id']!=identifier or record['selected_sha256']!=core.file_sha(folder/'deployed.pt'):
            raise RuntimeError('Existing worker did not finish successfully')
        return
    if adopt_pid:
        while True:
            args=proc_args(adopt_pid)
            if not args:break
            if 'code/scripts/train_oapa.py' not in args or output_argument(args)!=job['run_dir']:
                raise RuntimeError('Adopted PID identity changed')
            time.sleep(2)
        if not (folder/'run.json').exists():
            raise RuntimeError('Adopted trainer exited without run.json; inspect its original log before resuming')
    core.train(job)
    core.select(job);core.calibrate(job);core.compact(job)
    core.write_json(folder/'parallel_worker_complete.json',{'run_id':identifier,'finished_utc':now(),
        'selected_sha256':core.file_sha(folder/'deployed.pt'),'original_trainer_adopted':adopt_pid,
        'unchanged_protocol_sha256':core.file_sha(core.PACKAGE/'protocol.json')})


def parity_check():
    """Compare two concurrent short runs against the retained serial M0 weights."""
    from oapa4r_preflight import smoke_specs
    folder=core.OUT/'parallel_acceptance';folder.mkdir(parents=True,exist_ok=True)
    specs=smoke_specs(core);running=[]
    for condition in ('SV','PF'):
        spec=next(s for s in specs if s['condition']==condition and s['variant']=='F111' and s['p'] is None)
        target=folder/condition;argv=list(spec['argv']);argv[argv.index('--output')+1]=core.rel(target)
        if (target/'run.json').exists():running.append((None,None,spec,target));continue
        if target.exists() and any(target.iterdir()):argv+=['--resume']
        log=(folder/(condition+'.log')).open('a')
        process=subprocess.Popen([sys.executable]+argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        running.append((process,log,spec,target))
    results=[]
    for process,log,spec,target in running:
        if process:
            code=process.wait();log.close()
            if code:raise RuntimeError('Concurrent parity training failed: '+spec['condition'])
        left=core.torch.load(ROOT/spec['run_dir']/'model.pt',map_location='cpu',weights_only=True)
        right=core.torch.load(target/'model.pt',map_location='cpu',weights_only=True)
        max_parameter_difference=max(float((value-right['model'][name]).abs().max()) for name,value in left['model'].items())
        for name,value in left['model'].items():core.torch.testing.assert_close(value,right['model'][name],rtol=0,atol=1e-6)
        with core.np.load(ROOT/spec['run_dir']/'D_val_predictions.npz') as a,core.np.load(target/'D_val_predictions.npz') as b:
            fields=[k for k in a.files if k not in ('ids','labels','groups')]
            max_probability_difference=max(float(core.np.max(core.np.abs(a[k]-b[k]))) for k in fields)
            for key in fields:
                core.np.testing.assert_allclose(a[key],b[key],rtol=0,atol=1e-6)
                core.np.testing.assert_array_equal(a[key]>.5,b[key]>.5)
        old=core.read(ROOT/spec['run_dir']/'run.json')['selection'];new=core.read(target/'run.json')['selection']
        if {k:v['epoch'] for k,v in old.items()}!={k:v['epoch'] for k,v in new.items()}:
            raise RuntimeError('Concurrent acceptance changed stage checkpoint selection')
        for path in target.glob('mask_schedule_*.npz'):
            with core.np.load(path) as a,core.np.load(ROOT/spec['run_dir']/path.name) as b:
                for key in a.files:core.np.testing.assert_array_equal(a[key],b[key])
        results.append({'condition':spec['condition'],'state_dict_exact':max_parameter_difference==0,
            'max_parameter_abs_difference':max_parameter_difference,
            'max_validation_probability_abs_difference':max_probability_difference,
            'masks_exact':True,'validation_decisions_exact':True,'stage_selection_epochs_equal':True})
    core.write_json(core.OUT/'parallel_acceptance.json',{'passed':True,'runs':results,
        'purpose':'Paired concurrency acceptance: exact seed/order/masks/validation decisions, FP32 absolute parameter/probability tolerance 1e-6; not a bitwise determinism claim',
        'absolute_tolerance':1e-6,
        'time_utc':now(),'D_test':'not_predicted'})


def adopt_serial():
    global FOLLOW_WORKERS
    state=core.read(core.OUT/'queue_state.json')
    pid=state.get('pid');args=proc_args(pid) if pid else []
    allowed={'code/scripts/oapa4r_main.py'}
    if args and not any(x in allowed for x in args):
        raise RuntimeError('Scheduler identity mismatch; refusing to stop')
    # Preserve the full previous scope and its running worker identities.
    history=core.OUT/'scope_handoff_p0_p1_20261031.json'
    if not history.exists():core.write_json(history,{'previous_state':state,'time_utc':now()})
    valid={j['id']:j for j in core.JOBS}
    for identifier,entry in state['jobs'].items():
        if entry['status']!='running':continue
        if identifier not in valid:raise RuntimeError('Out-of-scope worker is active; explicit handling required')
        worker_pid=entry.get('worker_pid');worker_args=proc_args(worker_pid) if worker_pid else []
        if worker_args:
            if '--worker-run' not in worker_args or worker_args[worker_args.index('--worker-run')+1]!=identifier:
                raise RuntimeError('Worker identity mismatch')
            FOLLOW_WORKERS[valid[identifier]['run_dir']]=worker_pid
    if args:
        os.kill(pid,signal.SIGTERM)
        for _ in range(100):
            if not proc_args(pid):break
            time.sleep(.1)
        else:raise RuntimeError('Old scheduler still active')
    state=core.read(core.OUT/'queue_state.json')
    removed=[identifier for identifier in state['jobs'] if identifier not in valid]
    state['jobs']={k:v for k,v in state['jobs'].items() if k in valid}
    state['excluded_job_ids']=removed
    state['scope']='P0/P1 only, user requested 2026-10-31'
    state.pop('error',None)
    core.write_json(core.OUT/'queue_state.json',state)
    core.write_json(core.OUT/'active_scope.json',{'formal_jobs':72,'blocks':{'MAIN_P0':9,'MAIN_P1':63},
        'excluded_jobs':removed,'training_job_ids':[j['id'] for j in core.JOBS],
        'original_protocol_preserved':True,'time_utc':now(),
        'exports':['p0_sv_tt','p0_pf','p1'],'test_policy':'Freeze all 72 jobs before new test scoring'})
    live=trainers()
    for directory in FOLLOW_WORKERS:live.pop(directory,None)
    return live


def schedule(limit,adopt=False):
    import fcntl
    if not core.read(core.OUT/'parallel_acceptance.json').get('passed'):raise RuntimeError('Measured concurrency acceptance required')
    readiness=core.read(core.OUT/'readiness.json')
    if not readiness['ready'] or readiness['protocol_sha256']!=core.file_sha(core.PACKAGE/'protocol.json'):
        raise RuntimeError('M0/protocol identity mismatch')
    core.verify_hashes(readiness['source_hashes']);core.verify_hashes(readiness['data_hashes'])
    existing=adopt_serial() if adopt else trainers()
    lock=(core.OUT/'campaign.lock').open('a')
    deadline=time.monotonic()+30
    while True:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
        except BlockingIOError:
            if time.monotonic()>=deadline:raise RuntimeError('Campaign ownership lock did not become available')
            time.sleep(.5)
    state=core.read(core.OUT/'queue_state.json');active={};telemetry=[];failed=False
    manifest=core.OUT/'parallel_orchestration.json'
    if not manifest.exists():core.write_json(manifest,{'mode':'same_gpu_multiprocess','max_workers':limit,
        'worker_cpu_threads':4,'min_free_gpu_mib':8192,'scientific_sources_changed':False,
        'baseline_before_switch':{'gpu_utilization_percent':[21,23,23,11,24],'memory_used_mib':1555},
        'parity':'parallel_acceptance.json','time_utc':now()})
    core.write_json(core.OUT/'parallel_control.json',{'max_workers':limit})
    def update(status='TRAINING_PARALLEL',**kwargs):
        state.update(status=status,pid=os.getpid(),mode='parallel',parallel_limit=limit,
            current_jobs=list(active),current_job=next(iter(active)) if len(active)==1 else None,
            counts=dict(Counter(j['status'] for j in state['jobs'].values())),updated_utc=now(),**kwargs)
        core.write_json(core.OUT/'queue_state.json',state)
    blocks={'MAIN_P0':0,'MAIN_P1':1,'COMPONENTS':2,'SEMANTIC':3}
    jobs=sorted(core.JOBS,key=lambda j:blocks[j['block']])
    pending=[]
    for job in jobs:
        entry=state['jobs'][job['id']]
        if entry['status']=='completed':continue
        pending.append(job)
    def launch(job):
        entry=state['jobs'][job['id']];pid=existing.pop(job['run_dir'],None);follow=FOLLOW_WORKERS.pop(job['run_dir'],None)
        argv=[sys.executable,str(Path(__file__).resolve()),'--worker-run',job['id'],'--execute']
        if pid:argv+=['--adopt-trainer-pid',str(pid)]
        if follow:argv+=['--follow-worker-pid',str(follow)]
        logpath=core.OUT/'logs'/(job['id']+'_parallel_worker.log')
        stream=logpath.open('a');process=subprocess.Popen(argv,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        active[job['id']]={'process':process,'stream':stream,'job':job,'log':logpath}
        entry.update(status='running',worker_pid=process.pid,adopted_trainer_pid=pid,parallel_started_utc=now(),
                     attempts=entry.get('attempts',0)+1)
        print(json.dumps({'launched':job['id'],'worker_pid':process.pid,'adopted_trainer_pid':pid,'active':len(active)}),flush=True)
        update()
    try:
        # Adopt existing trainers first, even if their position in pending differs.
        for job in list(pending):
            if job['run_dir'] in existing or job['run_dir'] in FOLLOW_WORKERS:launch(job);pending.remove(job)
        if FOLLOW_WORKERS:raise RuntimeError('Unrecognized worker during handoff')
        if existing:raise RuntimeError('Unrecognized live trainers; refusing duplicate scheduling')
        while pending or active:
            requested=int(core.read(core.OUT/'parallel_control.json')['max_workers'])
            limit=max(1,min(8,requested))
            for identifier,item in list(active.items()):
                code=item['process'].poll()
                if code is None:continue
                item['stream'].close();entry=state['jobs'][identifier];job=item['job']
                del active[identifier]
                if code==0 and (ROOT/job['run_dir']/'parallel_worker_complete.json').exists():
                    entry.update(status='completed',completed_utc=now())
                    print(json.dumps({'completed':identifier,'remaining':len(pending)}),flush=True)
                else:
                    log=item['log'].read_text(errors='replace')[-65536:]
                    trainlog=core.OUT/'logs'/(identifier+'.log')
                    if trainlog.exists():log+=trainlog.read_text(errors='replace')[-65536:]
                    oom='out of memory' in log.lower()
                    if oom and entry['attempts']<3:
                        entry.update(status='pending',last_failure='OOM; retry same configuration after drain')
                        pending.insert(0,job);limit=max(1,limit-1)
                        core.write_json(core.OUT/'parallel_control.json',{'max_workers':limit})
                    else:
                        entry.update(status='failed',exit_code=code,error_log=core.rel(item['log']))
                        failed=True
                update()
            metric=gpu();metric['active_jobs']=len(active);telemetry.append(metric);telemetry=telemetry[-120:]
            core.write_json(core.OUT/'parallel_telemetry.json',{'samples':telemetry,'last':metric,
                'mean_utilization_percent':sum(m['utilization_percent'] for m in telemetry)/len(telemetry)})
            while pending and len(active)<limit and not failed:
                if metric['memory_free_mib']<8192:break
                if core.shutil.disk_usage(core.OUT).free<max(2,len(active)+1)*1024**3:break
                launch(pending.pop(0));metric=gpu()
            if failed and not active:raise RuntimeError('Worker failure; no new tasks launched, other workers drained')
            update();time.sleep(5)
        update('FREEZING')
        core.verify_hashes(readiness['source_hashes']);core.verify_hashes(readiness['data_hashes'])
        core.freeze()
        frozen=core.read(core.OUT/'frozen_manifest.json')
        frozen['files'].update(core.fingerprint([Path(__file__),Path(__file__).with_name('oapa4r_reporting.py'),manifest,core.OUT/'active_scope.json',core.OUT/'parallel_acceptance.json']))
        core.write_json(core.OUT/'frozen_manifest.json',frozen)
        core.verify_hashes(frozen['files'])
        setup()
        for index,job in enumerate(core.JOBS):
            update('FROZEN_TEST_EVALUATION',evaluated_jobs=index)
            core.verify_hashes(readiness['source_hashes'])
            cache=core.DATA[job['condition']]['cache'];core.verify_hashes({cache:readiness['data_hashes'][cache]})
            core.evaluate(job)
        update('INDEPENDENT_VERIFY',evaluated_jobs=len(core.JOBS))
        core.independent_verify();core.aggregate();core.statistics();core.export();update('COMPLETED')
    except BaseException as exc:
        update('FAILED_STOPPED',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
        raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--adopt-serial',action='store_true');parser.add_argument('--parity-check',action='store_true')
    parser.add_argument('--worker-run');parser.add_argument('--adopt-trainer-pid',type=int)
    parser.add_argument('--follow-worker-pid',type=int)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if not args.execute:print('PREVIEW ONLY');return
    if not 1<=args.workers<=8:parser.error('workers must be in 1..8')
    if args.worker_run:worker(args.worker_run,args.adopt_trainer_pid,args.follow_worker_pid)
    elif args.parity_check:parity_check()
    else:schedule(args.workers,args.adopt_serial)


if __name__=='__main__':main()

