"""Full OaPA campaign: fit-only experts, deployed J selection, gates and frozen evaluation.

Run the detached serial driver with --phase campaign. No test prediction is
permitted until every formal job, selector and calibration has completed.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
import joblib
import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from oapa.protocol import load_cache,file_sha,views,predict
from oapa.pipeline import (write_json,restore,extract,score,route,neural_table,
                           fit_gate,OaPAVPA,chosen_masks)

PACKAGE=ROOT/'experiments/oapa'
OUT=ROOT/'outputs/oapa'
PROTOCOL=json.loads((PACKAGE/'protocol.json').read_text(encoding='utf-8'))
JOBS=json.loads((PACKAGE/'training_jobs.json').read_text(encoding='utf-8'))
DATA=PROTOCOL['datasets']
DEVICE='cuda'
GATE_KINDS=('full','mask','confidence','agreement')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def rel(path):return Path(path).resolve().relative_to(ROOT).as_posix()


def fingerprint(paths):return {rel(p):file_sha(p) for p in sorted(set(map(Path,paths)))}


def verify_hashes(entries):
    for path,digest in entries.items():
        if not (ROOT/path).is_file() or file_sha(ROOT/path)!=digest:
            raise RuntimeError('Frozen input changed: '+path)


def runtime_sources():
    return list((ROOT/'code/oapa').glob('*.py'))+list((ROOT/'code/third_party').rglob('*.py'))+[
        Path(__file__),ROOT/'code/scripts/oapa_preflight.py',ROOT/'code/scripts/oapa_env.py',ROOT/'code/scripts/train_oapa.py',ROOT/'code/tests/test_oapa_experiments.py']


@lru_cache(maxsize=3)
def data(condition):return load_cache(ROOT/DATA[condition]['cache'])


def experts(condition):
    meta=read(OUT/'experts'/condition/'manifest.json');verify_hashes(meta['files'])
    return {v:joblib.load(ROOT/path) for v,path in meta['models'].items()}


def fit_experts(conditions=None):
    selected=set(conditions) if conditions is not None else set(DATA)
    for condition,spec in DATA.items():
        if condition not in selected:continue
        folder=OUT/'experts'/condition;manifest=folder/'manifest.json'
        if manifest.exists():verify_hashes(read(manifest)['files']);continue
        splits,dims,lengths,modalities,_=data(condition);batch=splits['D_fit']
        folder.mkdir(parents=True,exist_ok=True);models={};times={}
        for view in spec['expert_views']:
            tick=time.perf_counter();k=modalities.index(view)
            algorithm=(LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto') if view=='A'
                       else SVC(C=1,kernel='rbf',gamma='scale',probability=True,random_state=20261030))
            model=make_pipeline(StandardScaler(),algorithm)
            model.fit(batch.features[k].mean(1).numpy(),batch.labels.numpy())
            path=folder/f'{view}.joblib';joblib.dump(model,path);models[view]=rel(path)
            times[view]=time.perf_counter()-tick
        write_json(manifest,{'condition':condition,'fit_split':'D_fit','fit_n':len(batch),
            'fit_ids_sha256':hashlib.sha256(json.dumps(batch.ids).encode()).hexdigest(),
            'cache_sha256':file_sha(ROOT/spec['cache']),'models':models,
            'files':fingerprint(ROOT/p for p in models.values()),'seconds':times,
            'shared_across_seeds_and_p':True,'D_test':'not_predicted'})
        print(json.dumps({'experts_fitted':condition,'models':list(models)}),flush=True)


def selected_path(folder):
    manifest=read(folder/'selected_checkpoint.json');verify_hashes({manifest['path']:manifest['sha256']})
    return ROOT/manifest['path']


def select(job):
    folder=ROOT/job['run_dir'];dest=folder/'selected_checkpoint.json'
    if dest.exists():
        manifest=read(dest)
        if manifest.get('protocol_sha256')!=file_sha(PACKAGE/'protocol.json'):
            raise RuntimeError('Selected checkpoint predates the current selection protocol')
        selected_path(folder);return
    report=read(folder/'run.json');splits,dims,lengths,modalities,_=data(job['condition'])
    bank=experts(job['condition']);names=list(views(modalities));candidates=[]
    expected=report['recipe']['stages']['J']
    for epoch in range(1,expected+1):
        path=folder/f'J_epoch_{epoch}.pt'
        if not path.exists():raise FileNotFoundError(path)
        model=restore(path,report,DEVICE)
        values=extract(model,splits['D_val'],modalities,DEVICE)
        table=route(neural_table(values),names,splits['D_val'],modalities,bank)
        excluded_views=bank if job['p'] is None else ()
        result=score(table,splits['D_val'],modalities,job['p'],excluded_views=excluded_views)
        criterion=result['selection_score']
        candidates.append({'epoch':epoch,'score':criterion,'path':rel(path),'sha256':file_sha(path)})
        del model,values;torch.cuda.empty_cache() if DEVICE.startswith('cuda') else None
    best=max(candidates,key=lambda c:c['score'])
    deployed=folder/'deployed.pt';shutil.copyfile(ROOT/best['path'],deployed)
    if file_sha(deployed)!=best['sha256']:raise RuntimeError('Checkpoint copy failed')
    write_json(dest,{'path':rel(deployed),'sha256':file_sha(deployed),'epoch':best['epoch'],
                     'score':best['score'],'candidates':candidates,'tie':'earliest','p':job['p'],
                     'selection_split':'D_val','before_VPA':True,
                     'excluded_selection_views':result['excluded_views'],
                     'protocol_sha256':file_sha(PACKAGE/'protocol.json'),
                     'expert_manifest_sha256':file_sha(OUT/'experts'/job['condition']/'manifest.json')})


def calibrate(job):
    folder=ROOT/job['run_dir'];report=read(folder/'run.json')
    kinds=GATE_KINDS if job['block']=='MAIN_P0' and job['condition'] in ('SV','TT') else ('full',)
    if all((folder/f'gate_{k}.json').exists() for k in kinds):
        for k in kinds:verify_hashes(read(folder/f'gate_{k}.json')['files'])
        return
    splits,dims,lengths,modalities,_=data(job['condition']);split=DATA[job['condition']]['calibration']
    path=selected_path(folder);model=restore(path,report,DEVICE)
    values=extract(model,splits[split],modalities,DEVICE)
    for kind in kinds:
        target=folder/f'gate_{kind}.pt';gate,training=fit_gate(values,splits[split],modalities,job['p'],kind)
        torch.save({'model':gate.state_dict(),'kind':kind,'modalities':modalities,
                    'checkpoint_sha256':file_sha(path)},target)
        write_json(folder/f'gate_{kind}.json',{'files':fingerprint([target]),'training':training,
                       'checkpoint_sha256':file_sha(path),'protocol_sha256':file_sha(PACKAGE/'protocol.json'),
                       'D_test':'not_predicted'})
    del model,values
    if DEVICE.startswith('cuda'):torch.cuda.empty_cache()


def load_gate(folder,kind,m):
    meta=read(folder/f'gate_{kind}.json');verify_hashes(meta['files'])
    state=torch.load(folder/f'gate_{kind}.pt',map_location='cpu',weights_only=True)
    gate=OaPAVPA(m,kind);gate.load_state_dict(state['model'],strict=True)
    return gate.eval()


def compact(job,smoke=False):
    """User-approved retention: delete only ephemeral files inside this owned run."""
    folder=(ROOT/job['run_dir']).resolve()
    if not folder.is_relative_to(OUT.resolve()):raise ValueError('Retention target outside campaign')
    report=read(folder/'run.json')
    if job['method']=='OaPA' and not smoke:
        selected_path(folder)
        if not (folder/'gate_full.json').exists():raise RuntimeError('Do not compact before calibration')
    patterns=['stage_*.pt','J_epoch_*.pt','teachers_epoch_*.pt','resume.pt']
    removed=[]
    for pattern in patterns:
        for p in sorted(folder.glob(pattern)):
            if not p.is_file() or p.parent!=folder:raise ValueError('Unsafe retention path')
            removed.append({'name':p.name,'sha256':file_sha(p),'bytes':p.stat().st_size})
            p.unlink()
    if removed:
        write_json(folder/'retention.json',{'policy':'user_approved_selected_only_20261031',
                   'kept':['model.pt','deployed.pt if OaPA','selected_checkpoint.json','gates','logs','predictions','masks'],
                   'removed_ephemeral':removed,'raw_model_sha256':report['model_sha256']})


def train(job,extra=None,log_suffix=''):
    folder=ROOT/job['run_dir']
    if (folder/'run.json').exists():
        report=read(folder/'run.json')
        from train_oapa import implementation_hash
        if report['implementation_sha256']!=implementation_hash():raise RuntimeError('Completed run source identity differs; keep prior attempt and rerun in a new directory')
        if report['configuration_sha256']!=job['config_sha256']:raise RuntimeError('Config drift')
        if file_sha(folder/'model.pt')!=report['model_sha256']:raise RuntimeError('Raw model drift')
        return
    if shutil.disk_usage(OUT).free<2*1024**3:raise RuntimeError('Less than 2GiB free; stop before new task')
    argv=list(job['argv']);argv[argv.index('--device')+1]=DEVICE
    if folder.exists() and any(folder.iterdir()) and not extra:
        if job['method']=='OaPA' and (folder/'resume.pt').exists():argv+=['--resume']
        else:raise RuntimeError('Unfinished task requires explicit attempt recovery: '+job['id'])
    if extra:argv+=extra
    log=OUT/'logs'/(job['id']+log_suffix+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a',encoding='utf-8') as stream:
        subprocess.run([sys.executable]+argv,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)


def train_cell(run_id):
    matches=[job for job in JOBS if job['id']==run_id]
    if len(matches)!=1:
        raise ValueError('Unknown or duplicate run id: '+run_id)
    job=matches[0]
    fit_experts([job['condition']])
    train(job)
    select(job)
    calibrate(job)
    print(json.dumps({'trained':job['id'],'run_dir':job['run_dir'],'D_test':'not_evaluated'}),flush=True)


def test_cell(run_id):
    matches=[job for job in JOBS if job['id']==run_id]
    if len(matches)!=1:
        raise ValueError('Unknown or duplicate run id: '+run_id)
    job=matches[0]
    if job['method']!='OaPA':
        raise ValueError('test-cell supports the OaPA model only')
    folder=ROOT/job['run_dir'];metrics_path=folder/'test_metrics.json';manifest_path=folder/'test_manifest.json'
    predictions_path=folder/'test_predictions.npz'
    if metrics_path.exists():
        existing=read(metrics_path)
        if existing.get('manifest_sha256')!=file_sha(manifest_path):
            raise RuntimeError('Test artifacts belong to another frozen cell state')
        verify_hashes(read(manifest_path)['files'])
        verify_hashes(existing['prediction_files'])
        return
    report=read(folder/'run.json')
    selected=selected_path(folder);selected_meta=folder/'selected_checkpoint.json'
    gate_pt=folder/'gate_full.pt';gate_json=folder/'gate_full.json'
    if not gate_pt.is_file() or not gate_json.is_file():
        raise FileNotFoundError('Missing VPA full-gate artifacts')
    bank=experts(job['condition']);expert_manifest=OUT/'experts'/job['condition']/'manifest.json'
    expert_files=list(read(expert_manifest)['models'].values())
    inputs=[PACKAGE/'protocol.json',ROOT/DATA[job['condition']]['cache'],folder/'run.json',folder/'model.pt',
            selected_meta,selected,gate_pt,gate_json,expert_manifest]+[ROOT/p for p in expert_files]
    write_json(manifest_path,{'run_id':job['id'],'selection_split':'D_val','test_split':'D_test',
                              'files':fingerprint(inputs),'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    splits,dims,lengths,modalities,_=data(job['condition']);batch=splits['D_test']
    model=restore(selected,report,DEVICE);values=extract(model,batch,modalities,DEVICE);names=list(values)
    gate=load_gate(folder,'full',len(modalities))
    table=route(neural_table(values,gate),names,batch,modalities,bank)
    result=score(table,batch,modalities,job['p'])
    extra={}
    if job['p'] is not None:
        chosen,masks=chosen_masks(batch,modalities,job['p']);extra.update(chosen=chosen,masks=masks)
    temporary=predictions_path.with_suffix('.tmp.npz')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,ids=np.asarray(batch.ids),groups=np.asarray(batch.groups),
                            events=np.asarray(batch.events),labels=batch.labels.numpy(),
                            views=np.asarray(names),probabilities=table,**extra)
    temporary.replace(predictions_path)
    write_json(metrics_path,{'run_id':job['id'],'condition':job['condition'],'method':job['method'],
                             'variant':job['variant'],'seed':job['seed'],'p':job['p'],'result':result,
                             'manifest_sha256':file_sha(manifest_path),
                             'prediction_files':fingerprint([predictions_path]),'D_test':'evaluated'})
    del model,values
    if DEVICE.startswith('cuda'):torch.cuda.empty_cache()
    print(json.dumps({'tested':job['id'],'metrics':rel(metrics_path)},flush=True))


def freeze():
    inputs=[PACKAGE/'protocol.json',PACKAGE/'training_jobs.json',PACKAGE/'paper_output_schema.json']+runtime_sources()
    for d in DATA.values():inputs.append(ROOT/d['cache'])
    inputs+=list((OUT/'experts').rglob('*joblib'))+list((OUT/'experts').rglob('manifest.json'))
    for job in JOBS:
        folder=ROOT/job['run_dir'];report=read(folder/'run.json')
        if report['fit_limit'] is not None or report['smoke_override'] is not None:raise RuntimeError('Smoke in formal matrix')
        if report['D_test']!='not_evaluated':raise RuntimeError('Training test leakage')
        if (report['seed'],report['method'],report['variant'],report['p'])!=(job['seed'],job['method'],job['variant'],job['p']):raise RuntimeError('Run identity mismatch')
        if report['configuration_sha256']!=file_sha(ROOT/job['config_path']):raise RuntimeError('Configuration changed')
        if report['cache_sha256']!=file_sha(ROOT/DATA[job['condition']]['cache']):raise RuntimeError('Training cache changed')
        inputs += [folder/'run.json',folder/'model.pt',ROOT/job['config_path']]
        inputs += list(folder.glob('mask_schedule_*.npz'))
        if job['method']=='OaPA':
            inputs += [selected_path(folder),folder/'selected_checkpoint.json']
            inputs += list(folder.glob('gate_*.pt'))+list(folder.glob('gate_*.json'))
    inputs+=list((OUT/'masks').glob('*.npz'))
    comparisons={}
    for c in DATA:
        for block in ('MAIN_P0','MAIN_P1'):
            candidates={}
            for j in JOBS:
                if j['condition']!=c or j['block']!=block or j['method']=='OaPA':continue
                report=read(ROOT/j['run_dir']/'run.json')
                candidates.setdefault(j['method'],[]).append(report['D_val']['selection_score'])
            comparisons[c+'|'+block]={'method':max(candidates,key=lambda k:float(np.mean(candidates[k]))) if candidates else None,
                'D_val_scores':{k:float(np.mean(v)) for k,v in candidates.items()},'selection_split':'D_val'}
    comparison_path=OUT/'comparison_baselines.json'
    if not comparison_path.exists():write_json(comparison_path,comparisons)
    inputs.append(comparison_path)
    target=OUT/'frozen_manifest.json'
    if target.exists():verify_hashes(read(target)['files']);return
    write_json(target,{'status':'FROZEN_BEFORE_NEW_TEST_SCORING','files':fingerprint(inputs),
                       'formal_jobs':len(JOBS),'historical_test_exposure':True,'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})


def evaluate(job):
    # Full-campaign hashes are checked by the driver before entering this phase.
    frozen=read(OUT/'frozen_manifest.json')
    folder=ROOT/job['run_dir'];dest=folder/'paper_evaluation.json'
    if dest.exists():
        existing=read(dest)
        if existing['freeze_sha256']!=file_sha(OUT/'frozen_manifest.json'):raise RuntimeError('Evaluation belongs to another freeze')
        verify_hashes(existing['prediction_files']);return
    report=read(folder/'run.json');splits,dims,lengths,modalities,_=data(job['condition'])
    path=selected_path(folder) if job['method']=='OaPA' else folder/'model.pt'
    verify_hashes({rel(path):frozen['files'][rel(path)]})
    batch=splits['D_test'];model=restore(path,report,DEVICE)
    values=extract(model,batch,modalities,DEVICE);names=list(values)
    bank=experts(job['condition']) if job['method']=='OaPA' else {}
    gate=load_gate(folder,'full',len(modalities)) if bank else None
    table=route(neural_table(values,gate),names,batch,modalities,bank)
    result=score(table,batch,modalities,job['p'])
    extra={}
    if job['p'] is not None:
        chosen,masks=chosen_masks(batch,modalities,job['p']);extra.update(chosen=chosen,masks=masks)
    predpath=folder/'paper_predictions.npz'
    np.savez_compressed(predpath,ids=np.asarray(batch.ids),groups=np.asarray(batch.groups),events=np.asarray(batch.events),
                       labels=batch.labels.numpy(),views=np.asarray(names),probabilities=table,**extra)
    aggregation={}
    if job['method']=='OaPA' and job['block']=='MAIN_P0' and job['condition'] in ('SV','TT'):
        arms={'Backbone':neural_table(values),'Fixed alpha=.25':neural_table(values,alpha=.25),
              'Fixed alpha=.5':neural_table(values,alpha=.5)}
        for kind,label in [('mask','Mask only'),('confidence','Confidence only'),('agreement','Agreement only')]:
            arms[label]=neural_table(values,load_gate(folder,kind,len(modalities)))
        arms['OaPA']=table
        stored={}
        for label,arm in arms.items():
            routed=route(arm,names,batch,modalities,bank)
            aggregation[label]=score(routed,batch,modalities)
            stored[label]=routed
        np.savez_compressed(folder/'aggregation_predictions.npz',labels=batch.labels.numpy(),views=np.asarray(names),**stored)
        write_json(folder/'aggregation_evaluation.json',aggregation)
    paths=[predpath]+([folder/'aggregation_predictions.npz'] if aggregation else [])
    write_json(dest,{'run_id':job['id'],'condition':job['condition'],'method':job['method'],'variant':job['variant'],
                     'seed':job['seed'],'p':job['p'],'result':result,'aggregation':aggregation,
                     'freeze_sha256':file_sha(OUT/'frozen_manifest.json'),'prediction_files':fingerprint(paths)})
    del model,values
    if DEVICE.startswith('cuda'):torch.cuda.empty_cache()


def independent_verify():
    checks=[]
    for job in JOBS:
        folder=ROOT/job['run_dir'];result=read(folder/'paper_evaluation.json')
        verify_hashes(result['prediction_files'])
        with np.load(folder/'paper_predictions.npz',allow_pickle=False) as raw:
            y=raw['labels'];table=raw['probabilities'];names=raw['views'].tolist()
            per=[f1_score(y,table[:,i]>.5,labels=[0,1],average='macro',zero_division=0) for i in range(len(names))]
            np.testing.assert_allclose(per,[result['result']['views'][v]['macro_f1'] for v in names],atol=1e-12,rtol=0)
            np.testing.assert_allclose(np.mean(per),result['result']['F_all'],atol=1e-12,rtol=0)
            if job['p'] is not None:
                independent=[f1_score(y,table[np.arange(len(y)),c]>.5,labels=[0,1],average='macro',zero_division=0) for c in raw['chosen']]
                np.testing.assert_allclose(independent,result['result']['repetitions'],atol=1e-12,rtol=0)
        for row,values in result.get('aggregation',{}).items():
            with np.load(folder/'aggregation_predictions.npz',allow_pickle=False) as raw:
                per=[f1_score(raw['labels'],raw[row][:,i]>.5,labels=[0,1],average='macro',zero_division=0) for i in range(len(raw['views']))]
                np.testing.assert_allclose(np.mean(per),values['F_all'],atol=1e-12,rtol=0)
                if row=='OaPA':np.testing.assert_array_equal(raw[row],table)
        checks.append({'run_id':job['id'],'independent_sklearn_metric_match':True})
    coverage=[]
    for condition in ('SV','TT'):
        for seed in PROTOCOL['training_seeds']:
            pair=[next(j for j in JOBS if j['condition']==condition and j['seed']==seed and j['block']==block and j['method']=='OaPA') for block in ('MAIN_P0','SEMANTIC')]
            records=[]
            for j in pair:
                records.append([{'stage':r['stage'],'epoch':r['epoch'],'coverage':r['dcr_coverage'],'views':r['view_counts']}
                                for line in (ROOT/j['run_dir']/'epochs.jsonl').read_text().splitlines()
                                if (r:=json.loads(line))['stage'] in ('B','J')])
            if records[0]!=records[1]:raise RuntimeError('DCR usage or view count mismatch: '+condition)
            coverage.append({'condition':condition,'seed':seed,'matched':True,'records':records[0]})
    write_json(OUT/'independent_audit.json',{'passed':True,'formal_jobs':len(checks),'checks':checks,'paired_DCR':coverage})


def aggregate():
    if not read(OUT/'independent_audit.json')['passed']:raise RuntimeError('Audit required')
    results={j['id']:read(ROOT/j['run_dir']/'paper_evaluation.json') for j in JOBS}
    schema=read(PACKAGE/'paper_output_schema.json');tables={}
    def runs(block,c,method='OaPA',variant='F111',p=None):
        return [results[j['id']] for j in JOBS if (j['block'],j['condition'],j['method'],j['variant'],j['p'])==(block,c,method,variant,p)]
    def mean(values):
        if len(values)!=3:raise ValueError('Exactly three seeds required')
        return {'mean':float(np.mean(values)),'seed_std':float(np.std(values,ddof=1)),'seed_scores':values}
    for table in ('p0_sv_tt','p0_pf'):
        spec=schema[table];rows={}
        for c in spec['datasets']:
            for method in spec.get('rows_per_dataset',spec.get('rows',[])):
                rows[c+'|'+method]={v:None if c=='PF' and method=='MyGO_adapt' else
                    mean([r['result']['views'][v]['macro_f1'] for r in runs('MAIN_P0',c,method)]) for v in spec['columns']}
        tables[table]=rows
    tables['p1']={}
    for c in DATA:
        for method in schema['p1']['rows_per_dataset']:
            tables['p1'][c+'|'+method]={p:None if c=='PF' and method=='MyGO_adapt' else
                mean([r['result']['mean_macro_f1'] for r in runs('MAIN_P1',c,method,p=float(p))]) for p in schema['p1']['columns']}
    tables['components']={}
    for variant in schema['components']['rows']:
        row={}
        for condition in schema['components']['columns']:
            c,v=condition.split('/')
            rs=runs('MAIN_P0' if variant=='F111' else 'COMPONENTS',c,variant=variant)
            row[condition]=mean([r['result']['F_all'] if v=='F_all' else r['result']['views'][v]['macro_f1'] for r in rs])
        tables['components'][variant]=row
    tables['aggregation']={}
    for label in schema['aggregation']['rows']:
        row={}
        for condition in schema['aggregation']['columns']:
            c,v=condition.split('/');rs=runs('MAIN_P0',c)
            row[condition]=mean([r['aggregation'][label]['F_all'] if v=='F_all' else r['aggregation'][label]['views'][v]['macro_f1'] for r in rs])
        tables['aggregation'][label]=row
    tables['reference_selection']={label:{c:mean([r['result']['F_all'] for r in runs(block,c,variant=variant)]) for c in ('SV','TT')}
        for label,block,variant in [('S-random','SEMANTIC','S-random'),('Semantic (DCR)','MAIN_P0','F111')]}
    for c in ('SV','TT'):
        a=tables['components']['F111'][c+'/F_all']['seed_scores'];b=tables['aggregation']['OaPA'][c+'/F_all']['seed_scores']
        if a!=b:raise RuntimeError('Cross-table F_all mismatch')
    write_json(OUT/'aggregate.json',{'tables':tables,'schema':schema,'formal_jobs':len(JOBS),'three_seed_means':True,
                                    'external_baselines_evaluated':False})


def bootstrap_difference(y,groups,a,b,rng,repeats=5000):
    """Paired event bootstrap; rows represent fixed seeds/views/mask repeats."""
    y=np.asarray(y,dtype=bool);a=np.asarray(a,dtype=bool);b=np.asarray(b,dtype=bool)
    unique,inverse=np.unique(groups,return_inverse=True);g=len(unique);distribution=[]
    def means(pred,weights):
        tp=(pred & y).astype(float)@weights.T
        tn=((~pred)&(~y)).astype(float)@weights.T
        positives=y.astype(float)@weights.T;negatives=(~y).astype(float)@weights.T
        predicted=pred.astype(float)@weights.T;totals=weights.sum(1)
        d1=positives[None]+predicted;d0=negatives[None]+totals[None]-predicted
        f1=np.divide(2*tp,d1,out=np.zeros_like(tp),where=d1>0)
        f0=np.divide(2*tn,d0,out=np.zeros_like(tn),where=d0>0)
        return ((f1+f0)/2).mean(0)
    for start in range(0,repeats,128):
        indices=rng.integers(0,g,size=(min(128,repeats-start),g))
        counts=np.stack([np.bincount(row,minlength=g) for row in indices])
        weights=counts[:,inverse].astype(float)
        distribution.extend((means(a,weights)-means(b,weights)).tolist())
    observed=float((means(a,np.ones((1,len(y))))-means(b,np.ones((1,len(y)))))[0])
    return {'paired_event_bootstrap_repeats':repeats,'observed_delta_pp':observed*100,'ci95_pp':(np.quantile(distribution,[.025,.975])*100).tolist(),
            'bootstrap_std_pp':float(np.std(distribution,ddof=1)*100),'groups':g,
            'scope':'Event resampling conditional on three trained seeds; seed SD reported separately'}


def statistics():
    tables=read(OUT/'aggregate.json')['tables'];comparison=read(OUT/'comparison_baselines.json')
    if all(j['method']=='OaPA' for j in JOBS):
        rng=np.random.default_rng(20261031);comparisons={}
        for c in ('SV','TT'):
            for label,block,variant in [('components_vs_F000','COMPONENTS','F000'),('semantic_vs_random','SEMANTIC','S-random'),('VPA_vs_Backbone','MAIN_P0','F111')]:
                arrays=[];metadata=None
                for arm in ('full','control'):
                    rows=[]
                    for seed in PROTOCOL['training_seeds']:
                        job=next(j for j in JOBS if j['condition']==c and j['seed']==seed and j['method']=='OaPA' and
                            j['block']==('MAIN_P0' if arm=='full' else block) and j['variant']==('F111' if arm=='full' else variant))
                        folder=ROOT/job['run_dir']
                        with np.load(folder/'paper_predictions.npz',allow_pickle=False) as raw:
                            y=raw['labels'];groups=raw['events'];ids=raw['ids']
                            if metadata is None:metadata=(y.copy(),groups.copy(),ids.copy())
                            else:
                                for a,b in zip((y,groups,ids),metadata):np.testing.assert_array_equal(a,b)
                            table=raw['probabilities']
                        if arm=='control' and label=='VPA_vs_Backbone':
                            with np.load(folder/'aggregation_predictions.npz',allow_pickle=False) as raw:table=raw['Backbone']
                        rows.extend((table.T>.5).tolist())
                    arrays.append(np.asarray(rows))
                comparisons[c+'|'+label]=bootstrap_difference(metadata[0],metadata[1],arrays[0],arrays[1],rng)
        write_json(OUT/'statistics.json',{'seed':20261031,'within_OaPA_controls':comparisons,
            'external_baselines_evaluated':False,'external_win_counts':None,
            'scope':'OaPA and its planned controls only; existing baseline paper values unchanged; no new external superiority claim'})
        return
    wins={'P0':{},'P1':{}};bootstrap={}
    rng=np.random.default_rng(20261031)
    for c in DATA:
        p0=tables['p0_pf' if c=='PF' else 'p0_sv_tt'];ours=p0[c+'|OaPA']
        for v,value in ours.items():
            competitors=[row[v]['mean'] for key,row in p0.items() if key.startswith(c+'|') and not key.endswith('|OaPA') and row[v] is not None]
            wins['P0'][c+'/'+v]={'strict_win':value['mean']>max(competitors),'tie':value['mean']==max(competitors),
                                  'margin_pp':100*(value['mean']-max(competitors))}
        for p,value in tables['p1'][c+'|OaPA'].items():
            competitors=[row[p]['mean'] for key,row in tables['p1'].items() if key.startswith(c+'|') and not key.endswith('|OaPA') and row[p] is not None]
            wins['P1'][c+'/'+p]={'strict_win':value['mean']>max(competitors),'tie':value['mean']==max(competitors),
                                  'margin_pp':100*(value['mean']-max(competitors))}
        for block in ('MAIN_P0','MAIN_P1'):
            baseline=comparison[c+'|'+block]['method'];arrays={};metadata=None
            for method in ('OaPA',baseline):
                parts=[]
                for job in JOBS:
                    if (job['condition'],job['block'],job['method'])!=(c,block,method):continue
                    with np.load(ROOT/job['run_dir']/'paper_predictions.npz',allow_pickle=False) as raw:
                        table=raw['probabilities'];y=raw['labels'];groups=raw['events'];ids=raw['ids']
                        if metadata is None:metadata=(y.copy(),groups.copy(),ids.copy())
                        else:
                            for x,z in zip((y,groups,ids),metadata):np.testing.assert_array_equal(x,z)
                        if block=='MAIN_P0':parts.extend((table.T>.5).tolist())
                        else:parts.extend((table[np.arange(len(y))[None],raw['chosen']]>.5).tolist())
                arrays[method]=np.asarray(parts)
            bootstrap[c+'|'+block]={'baseline_selected_on_D_val':baseline,
                **bootstrap_difference(metadata[0],metadata[1],arrays['OaPA'],arrays[baseline],rng)}
    write_json(OUT/'statistics.json',{'seed':20261031,'wins':wins,'primary_comparisons':bootstrap,
        'P0_strict_wins':sum(x['strict_win'] for x in wins['P0'].values()),
        'P1_strict_wins':sum(x['strict_win'] for x in wins['P1'].values()),
        'historical_test_exposure':True,'all_claims_require_reading_full_condition_results':True})


def table_text(name,values,schema):
    spec=schema[name];columns=spec['columns']
    names={'UMGTN_Src_adapt':'UMGTN (2022)','CLKD_IMRD_adapt':'CLKD (2023)',
           'ShaSpec_adapt':'ShaSpec (2023)','MPLMM_adapt':'MPLMM (2024)',
           'MyGO_adapt':'MyGO (2026)','OaPA':'OaPA'}
    def cell(v,bold=False):
        if v is None:return '--'
        text=f"{v['mean']*100:.2f}"
        return r'\textbf{'+text+'}' if bold else text
    rows=[];header=[]
    if name in ('p0_sv_tt','p0_pf','p1'):
        datasets=spec['datasets'];methods=spec.get('rows_per_dataset',spec.get('rows'))
        header=[('Method' if name=='p0_pf' else 'Dataset & Method')+' & '+' & '.join(columns)+r'\\']
        alignment=('l' if name=='p0_pf' else 'll')+'r'*len(columns)
        for ci,c in enumerate(datasets):
            if ci:rows.append(r'\midrule')
            for method in methods:
                key=c+'|'+method
                if key not in values:raise ValueError('Missing table row '+key)
                row=values[key];cells=[]
                for v in columns:
                    item=row[v]
                    others=[values[c+'|'+m][v]['mean'] for m in methods if m!='OaPA' and values[c+'|'+m][v] is not None]
                    bold=method=='OaPA' or (item is not None and item['mean']==max(others))
                    if item is None and (c,method)!=('PF','MyGO_adapt'):raise ValueError('Unexpected missing result')
                    cells.append(cell(item,bold))
                label=r'\textbf{OaPA}' if method=='OaPA' else names[method]
                prefix=([c] if name!='p0_pf' else [])+[label]
                line=' & '.join(prefix+cells)+r'\\'
                rows.append((r'\rowcolor{gray!10} ' if method=='OaPA' else '')+line)
    elif name=='components':
        alignment='ccc'+'r'*len(columns)
        header=[r'MPD & VPC & DCR & SV/VA & TT/TA & SV/$F_{\rm all}$ & TT/$F_{\rm all}$\\']
        for variant in spec['rows']:
            row=values[variant];rows.append(' & '.join(list(variant[1:])+[cell(row[v],variant=='F111') for v in columns])+r'\\')
    elif name=='aggregation':
        alignment='lrrrrrr'
        header=[r'Method & \multicolumn{3}{c}{SV} & \multicolumn{3}{c}{TT}\\',
                r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',r' & TV & TA & $F_{\rm all}$ & TV & TA & $F_{\rm all}$\\']
        labels={'Fixed alpha=.25':r'Fixed $\alpha=.25$','Fixed alpha=.5':r'Fixed $\alpha=.5$'}
        for label in spec['rows']:
            rows.append(' & '.join([labels.get(label,label)]+[cell(values[label][v],label=='OaPA') for v in columns])+r'\\')
    else:
        alignment='lrr';header=[r'Reference selection & SV & TT\\']
        for label in spec['rows']:rows.append(' & '.join([label]+[cell(values[label][c]) for c in columns])+r'\\')
    caption={'p0_sv_tt':r'P0 per-view Macro-F1 (\%) on SV/TT.',
             'p0_pf':r'P0 per-view Macro-F1 (\%) on PF. A dash denotes an unavailable result.',
             'p1':r'P1 matched-dropout Macro-F1 (\%) on SV, TT, and PF. Each seed averages ten fixed masks.',
             'components':r'MPD/VPC/DCR ablations (Macro-F1, \%). VPA is retained and refitted for each backbone. $F_{\rm all}$ averages seven legal views.',
             'aggregation':r'Prediction aggregation comparisons (Macro-F1, \%). All rows share fixed singleton experts, including Backbone. $F_{\rm all}$ averages seven legal views.',
             'reference_selection':r'Semantic versus random reference selection (all-view Macro-F1, \%).'}[name]
    env='table' if name in ('p0_pf','reference_selection') else 'table*'
    text=['% Fresh campaign results; see aggregate.json and independent_audit.json.',
          '\\begin{'+env+'}[!htbp]',r'\centering\small','\\caption{'+caption+' Three-seed means.}',
          '\\label{tab:'+name+'}',r'\setlength{\tabcolsep}{4pt}',
          '\\begin{tabular}{'+alignment+'}',r'\toprule']+header+[r'\midrule']+rows+[
          r'\bottomrule',r'\end{tabular}','\\end{'+env+'}']
    return '\n'.join(text)+'\n'


def export():
    tables=read(OUT/'aggregate.json')['tables'];schema=read(PACKAGE/'paper_output_schema.json')
    dest=OUT/'export/tables';dest.mkdir(parents=True,exist_ok=True)
    for name,values in tables.items():
        (dest/(name+'.tex')).write_text(table_text(name,values,schema),encoding='utf-8')
    write_json(OUT/'completion.json',{'status':'COMPLETED','formal_training_jobs':len(JOBS),'full_gates':120,
        'restricted_gates':18,'experts':5,'export_files':fingerprint(dest.glob('*.tex')),
        'actual_paper_modified':False,'external_baselines_evaluated':False,'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})


def main():
    global DEVICE
    parser=argparse.ArgumentParser();parser.add_argument('--phase',required=True,
        choices=['train-cell','test-cell','fit-experts','select','calibrate','freeze','evaluate','verify','aggregate','export','preflight','campaign'])
    parser.add_argument('--run-id');parser.add_argument('--device',default='cuda');parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();DEVICE=args.device
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    if not args.execute:print('PREVIEW ONLY; add --execute to run '+args.phase);return
    OUT.mkdir(parents=True,exist_ok=True)
    if args.phase=='train-cell':
        if not args.run_id:parser.error('train-cell requires --run-id')
        train_cell(args.run_id);return
    if args.phase=='test-cell':
        if not args.run_id:parser.error('test-cell requires --run-id')
        test_cell(args.run_id);return
    if args.phase in ('preflight','campaign'):
        from oapa_preflight import preflight,campaign
        (preflight if args.phase=='preflight' else campaign)(sys.modules[__name__]);return
    if args.phase=='fit-experts':fit_experts();return
    if args.phase=='freeze':freeze();return
    if args.phase=='evaluate':verify_hashes(read(OUT/'frozen_manifest.json')['files'])
    if args.phase in ('select','calibrate','evaluate'):
        fn={'select':select,'calibrate':calibrate,'evaluate':evaluate}[args.phase]
        for job in JOBS:
            if args.run_id and job['id']!=args.run_id:continue
            if args.phase=='evaluate' or job['method']=='OaPA':fn(job)
    else:{'verify':independent_verify,'aggregate':aggregate,'export':export}[args.phase]()


if __name__=='__main__':main()
