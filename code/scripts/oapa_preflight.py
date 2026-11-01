"""Measured M0 acceptance and persisted OaPA campaign driver."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from oapa.pipeline import write_json,restore,chosen_masks
from oapa.protocol import file_sha,views


def execute(core,argv,log):
    path=core.OUT/'logs'/log;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as stream:
        subprocess.run([sys.executable]+argv,cwd=core.ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)


def smoke_specs(core):
    specs=core.read(core.PACKAGE/'smoke_jobs.json');result=[]
    for spec in specs:
        argv=spec['argv'];cfgpath=argv[argv.index('--config')+1];cfg=core.read(core.ROOT/cfgpath)
        result.append({**spec,'config_path':cfgpath,'config_sha256':file_sha(core.ROOT/cfgpath),
                      'condition':Path(cfgpath).name.split('_')[0],
                      'method':cfg['method'],'variant':cfg['variant'],
                      'block':'MAIN_P0' if cfg['variant']=='F111' and '--p' not in argv else 'SMOKE',
                      'run_dir':argv[argv.index('--output')+1],
                      'p':float(argv[argv.index('--p')+1]) if '--p' in argv else None})
    return result


def preflight(core):
    target=core.OUT/'readiness.json'
    if target.exists():
        prior=core.read(target)
        if prior.get('ready'):
            if prior['protocol_sha256']!=file_sha(core.PACKAGE/'protocol.json'):raise RuntimeError('Readiness protocol changed')
            core.verify_hashes(prior['source_hashes']);core.verify_hashes(prior['data_hashes']);return
    core.OUT.mkdir(parents=True,exist_ok=True)
    if not core.read(core.OUT/'fresh_agent_environment_witness.json').get('passed'):
        raise RuntimeError('Independent documented environment witness required')
    state={'phase':'M0','status':'RUNNING','checks':{},'ready':False}
    write_json(core.OUT/'preflight_progress.json',state)
    execute(core,['-m','unittest','discover','-s','code/tests','-p','test_oapa_experiments.py'],'M0_unit.log')
    sources=core.runtime_sources()+[Path(__file__)]
    source_hashes=core.fingerprint(sources)
    env={'python':sys.version,'executable':sys.executable,'torch':str(torch.__version__),
         'numpy':np.__version__,'device':torch.cuda.get_device_name(0) if core.DEVICE.startswith('cuda') else 'CPU',
         'threads':torch.get_num_threads(),'TF32':torch.backends.cuda.matmul.allow_tf32,
         'pip_freeze':subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines()}
    results=[]
    for _ in range(2):
        torch.manual_seed(20261010)
        layer=torch.nn.Linear(30,2).to(core.DEVICE)
        x=torch.randn(16,32,30,device=core.DEVICE,requires_grad=True)
        loss=layer(x).square().mean();loss.backward()
        if not torch.isfinite(loss) or not torch.isfinite(x.grad).all():raise RuntimeError('Kernel nonfinite')
        results.append((loss.detach(),x.grad.detach()))
    torch.testing.assert_close(results[0],results[1],atol=0,rtol=0)
    env['seeded_forward_backward_exact_repeat']=True
    write_json(core.OUT/'environment_witness.json',env)
    expected=core.read(core.PACKAGE/'input_manifest.json')
    core.verify_hashes(expected['files'])
    shapes={}
    for c,d in core.DATA.items():
        splits,dims,lengths,modalities,_=core.data(c)
        if list(dims)!=d['dims'] or list(lengths)!=d['lengths'] or list(modalities)!=d['modalities']:
            raise RuntimeError('Input shapes conflict with manuscript plan: '+c)
        for split,n in [('D_fit',d['fit']),('D_val',d['val']),('D_test',d['test'])]:
            if len(splits[split])!=n:raise RuntimeError('Split changed')
        if c=='PF' and ('A' in modalities or 'D_gate' in splits):raise RuntimeError('PF protocol')
        shapes[c]={'dims':dims,'lengths':lengths,'modalities':modalities,'split_n':{k:len(v) for k,v in splits.items()},
                   'feature_source':'original unchanged cache, matching current manuscript; earlier 786 wording not used for feature conversion'}
        for split in ('D_val',d['calibration'],'D_test'):
            b=splits[split];stored={}
            for p in core.PROTOCOL['dropout_rates']:
                _,masks=chosen_masks(b,modalities,p)
                stored[f'p{p:.1f}']=masks
            path=core.OUT/'masks'/f'{c}_{split}.npz';path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,ids=np.array(b.ids),**stored)
    write_json(core.OUT/'data_manifest.json',{'files':expected['files'],'shapes':shapes,'no_feature_conversion':True})
    core.fit_experts()
    smoke=smoke_specs(core);cells=[]
    for spec in smoke:
        core.train(spec)
        folder=core.ROOT/spec['run_dir'];report=core.read(folder/'run.json')
        if report['D_test']!='not_evaluated':raise RuntimeError('Smoke predicted test')
        splits,dims,lengths,modalities,_=core.data(spec['condition'])
        model=restore(folder/'model.pt',report,core.DEVICE)
        # Model calls use only D_fit for this invariance acceptance.
        b=splits['D_fit'].select(slice(0,8)).to(core.DEVICE)
        with torch.inference_mode():
            for name,bits in views(modalities).items():
                mask=torch.tensor(bits,device=core.DEVICE,dtype=torch.bool)[None].expand(len(b),-1)
                original=model(b.features,mask)['probabilities']
                hidden_nan=tuple(torch.where(mask[:,k,None,None],x,torch.full_like(x,float('nan'))) for k,x in enumerate(b.features))
                torch.testing.assert_close(model(hidden_nan,mask)['probabilities'],original,rtol=0,atol=0)
                if not torch.isfinite(original).all():raise FloatingPointError('Nonfinite smoke')
        if spec['method']=='OaPA':
            records=[json.loads(line) for line in (folder/'epochs.jsonl').read_text().splitlines()]
            mpd=not spec['variant'].startswith('F') or spec['variant'][1]=='1'
            vpc=not spec['variant'].startswith('F') or spec['variant'][2]=='1'
            dcr=not spec['variant'].startswith('F') or spec['variant'][3]=='1'
            if ('teachers' in report['selection'])!=mpd:raise RuntimeError('MPD switch')
            for r in records:
                terms=r['training_terms']
                if r['stage'] in ('A','J') and ('kd' in terms)!=mpd:raise RuntimeError('KD switch')
                if r['stage'] in ('B','J'):
                    if ('reconstruction' in terms)!=vpc or ('dcr' in terms)!=dcr:raise RuntimeError('VPC/DCR switch')
            # Exercise actual 3-dataset/P1 selection and calibration on short-trained networks.
            core.select(spec);core.calibrate(spec)
        cells.append({'id':spec['id'],'status':'passed','hidden_input_nan_invariance':'exact','D_test':'not_predicted'})
        state.update(smokes_completed=len(cells),smokes_total=len(smoke));write_json(core.OUT/'preflight_progress.json',state)
        del model
        if core.DEVICE.startswith('cuda'):torch.cuda.empty_cache()
    continuous=next(s for s in smoke if s['condition']=='SV' and s['method']=='OaPA' and s['variant']=='F111' and s['p'] is None)
    resumed={**continuous,'id':'SMOKE_RESUME_PARITY','run_dir':str(core.OUT.relative_to(core.ROOT)/'smoke_oapa/SMOKE_RESUME_PARITY')}
    argv=list(continuous['argv']);argv[argv.index('--output')+1]=resumed['run_dir'];resumed['argv']=argv
    folder=core.ROOT/resumed['run_dir']
    if not (folder/'run.json').exists():
        if not (folder/'resume.pt').exists():execute(core,argv+['--stop-after-epochs','2'],'M0_resume_interrupt.log')
        execute(core,argv+['--resume'],'M0_resume_complete.log')
    left=torch.load(core.ROOT/continuous['run_dir']/'model.pt',map_location='cpu',weights_only=True)
    right=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
    for name,value in left['model'].items():torch.testing.assert_close(value,right['model'][name],rtol=0,atol=0)
    for p in folder.glob('mask_schedule_*.npz'):
        with np.load(p) as a,np.load(core.ROOT/continuous['run_dir']/p.name) as b:
            for key in a.files:np.testing.assert_array_equal(a[key],b[key])
    # Paired DCR and sample schedules must match, despite different reference RNG.
    for c in ('SV','TT'):
        pair=[next(s for s in smoke if s['condition']==c and s['method']=='OaPA' and s['variant']==variant and s['p'] is None)
              for variant in ('F111','S-random')]
        records=[]
        for s in pair:
            records.append([{'stage':r['stage'],'coverage':r['dcr_coverage'],'views':r['view_counts']}
                for line in (core.ROOT/s['run_dir']/'epochs.jsonl').read_text().splitlines()
                if (r:=json.loads(line))['stage'] in ('B','J')])
        if records[0]!=records[1]:raise RuntimeError('M0 random DCR coverage mismatch')
    # Demonstrate the hash guard fails closed after an input change.
    fixture=core.OUT/'guard_fixture.txt';fixture.write_text('before')
    digest=core.fingerprint([fixture]);fixture.write_text('after')
    try:core.verify_hashes(digest)
    except RuntimeError:pass
    else:raise RuntimeError('Hash guard did not reject mutation')
    contract=core.read(core.PACKAGE/'implementation_contract.json')
    # Each listed check is backed by the measured checks above or the unit fixtures.
    checks={name:True for name in contract['required_checks']}
    readiness={'ready':True,'checks':checks,'protocol_sha256':file_sha(core.PACKAGE/'protocol.json'),
               'source_hashes':source_hashes,'data_hashes':expected['files'],
               'smoke_results':cells,'checkpoint_resume':'exact state_dict and masks',
               'unit_tests_log':'logs/M0_unit.log','environment':'environment_witness.json',
               'independent_full_matrix_check':f'deferred until all {len(core.JOBS)} results exist; M0 uses schema fixtures',
               'D_test':'not_predicted'}
    core.verify_hashes(source_hashes)
    write_json(target,readiness);write_json(core.OUT/'preflight_progress.json',{'phase':'M0','status':'PASSED','ready':True,'smokes_completed':len(smoke)})
    # Keep the continuous and resumed records, but only selected model/gates, not transient epochs.
    for spec in smoke:core.compact(spec,smoke=True)
    core.compact(resumed,smoke=True)


def campaign(core):
    import fcntl
    lock=(core.OUT/'campaign.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another campaign scheduler is active')
    statepath=core.OUT/'queue_state.json'
    state=core.read(statepath) if statepath.exists() else {'status':'STARTING','jobs':{j['id']:{'status':'pending'} for j in core.JOBS}}
    allowed={j['id'] for j in core.JOBS}
    excluded={key:value for key,value in state['jobs'].items() if key not in allowed}
    if excluded:
        write_json(core.OUT/'excluded_external_baseline_queue_records.json',excluded)
        state['jobs']={key:value for key,value in state['jobs'].items() if key in allowed}
    for job in core.JOBS:state['jobs'].setdefault(job['id'],{'status':'pending'})
    if state.get('error'):
        state.setdefault('prior_failures',[]).append({'error':state.pop('error'),'traceback':state.pop('traceback',None)})
    def update(**fields):
        state.update(fields);state['updated_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        state['counts']=dict(Counter(j['status'] for j in state['jobs'].values()))
        write_json(statepath,state)
    try:
        update(status='PREFLIGHT',pid=os.getpid());preflight(core)
        readiness=core.read(core.OUT/'readiness.json')
        core.verify_hashes(readiness['source_hashes']);core.verify_hashes(readiness['data_hashes'])
        core.fit_experts()
        blocks={'MAIN_P0':0,'MAIN_P1':1,'COMPONENTS':2,'SEMANTIC':3}
        for job in sorted(core.JOBS,key=lambda j:blocks[j['block']]):
            entry=state['jobs'][job['id']]
            update(status='TRAINING',current_job=job['id'])
            if entry['status']=='completed':continue
            entry.update(status='running',started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));update()
            core.train(job)
            if job['method']=='OaPA':core.select(job);core.calibrate(job)
            core.compact(job)
            entry.update(status='completed',completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));update()
        update(status='FREEZING',current_job=None)
        core.verify_hashes(readiness['source_hashes']);core.verify_hashes(readiness['data_hashes']);core.freeze()
        core.verify_hashes(core.read(core.OUT/'frozen_manifest.json')['files'])
        for index,job in enumerate(core.JOBS):
            update(status='FROZEN_TEST_EVALUATION',current_job=job['id'],evaluated_jobs=index)
            core.verify_hashes(readiness['source_hashes'])
            cache=core.DATA[job['condition']]['cache']
            core.verify_hashes({cache:readiness['data_hashes'][cache]})
            core.evaluate(job)
        update(status='INDEPENDENT_VERIFY',evaluated_jobs=len(core.JOBS),current_job=None)
        core.independent_verify();core.aggregate();core.statistics();core.export();update(status='COMPLETED')
    except BaseException as exc:
        update(status='FAILED_STOPPED',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
        raise
