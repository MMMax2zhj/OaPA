"""Train one OaPA cell; never predict D_test.

The independent evaluator is permitted only after a frozen selection manifest.
"""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from oapa.backbone import build_oapa_backbone
from oapa.layers import SequenceTeacher
from oapa.objectives import (
    DiscriminativeContrastiveRegularization,
    ModalityPrioritizedDistillation,
)
from oapa.protocol import binary_metrics,evaluate,file_sha,load_cache,sample_masks,views


RECIPE={'stages':{'teachers':10,'A':20,'B':10,'J':10},'batch_size':16,'optimizer':'AdamW','weight_decay':.01,'backbone_lr_scale':.1}
DEFAULT_MODEL={'hidden':128,'heads':4,'layers':2,'memory_layers':3,'encoder_layers':1,'dropout':.1,'prompt_length':8,'topics':8}


def implementation_hash():
    paths=sorted((ROOT/'code/oapa').glob('*.py'))+[Path(__file__)]
    return hashlib.sha256('\n'.join(p.relative_to(ROOT).as_posix()+':'+file_sha(p) for p in paths).encode()).hexdigest()


def backbone_loss(model,batch,mask,teachers,stage,variant,aux_scale,rng,view_ce_weights=None,modalities=None):
    out=model(batch.features,mask);m=mask.shape[1]
    if view_ce_weights:
        sample_weights=torch.ones(len(batch),device=mask.device,dtype=out['logits'].dtype)
        legal=views(modalities)
        for name,weight in view_ce_weights.items():
            matching=(mask==torch.as_tensor(legal[name],device=mask.device)).all(1)
            sample_weights=torch.where(matching,weight,sample_weights)
        ce=(F.cross_entropy(out['logits'],batch.labels,reduction='none')*sample_weights).sum()/sample_weights.sum()
    else:
        ce=F.cross_entropy(out['logits'],batch.labels)
    loss=ce;stats={'ce':float(ce.detach())};teacher_logits=None
    distill=not (variant.startswith('F') and variant[1]=='0')
    completion=not (variant.startswith('F') and variant[2]=='0')
    use_dcr=not (variant.startswith('F') and variant[3]=='0')
    if stage in ('A','J'):
        targets=batch.labels[:,None].expand(-1,m).flatten()
        branch=F.cross_entropy(out['branch_logits'].flatten(0,1),targets,reduction='none').reshape_as(mask)[mask].mean()
        loss=loss+.25*branch;stats['branch_ce']=float(branch.detach())
        if distill:
            with torch.no_grad():
                clean=tuple(torch.where(mask[:,k,None,None],x,torch.zeros_like(x)) for k,x in enumerate(batch.features))
                tf=torch.stack([t.projector(x) for t,x in zip(teachers,clean)],1)
                teacher_logits=torch.stack([t.head(tf[:,k]) for k,t in enumerate(teachers)],1)
            kd,feature=ModalityPrioritizedDistillation()(out['logits'],out['projected'],teacher_logits,tf,mask)
            loss=loss+aux_scale*(.1*kd+.1*feature)
            stats.update(kd=float(kd.detach()),feature_kd=float(feature.detach()))
    if stage in ('B','J'):
        if completion:
            errors=[]
            for k,(pred,target) in enumerate(zip(out['reconstructed'],batch.features)):
                chosen=~mask[:,k]
                if chosen.any(): errors.append(1-F.cosine_similarity(pred[chosen],target[chosen].mean(1).detach(),dim=-1))
            rec=torch.cat(errors).mean() if errors else ce*0
            loss=loss+aux_scale*.1*rec;stats['reconstruction']=float(rec.detach())
        if use_dcr:
            reference='random' if variant=='S-random' else 'semantic'
            dcr,coverage=DiscriminativeContrastiveRegularization(reference=reference)(
                out['representation'],batch.labels,batch.ids,batch.groups,rng=rng
            )
            coverage.pop('pairs')
            loss=loss+aux_scale*.1*dcr;stats.update(dcr=float(dcr.detach()),dcr_coverage=coverage)
    return loss,stats,teacher_logits,out


def save_checkpoint(path,model,method,settings,variant,stage,extra=None):
    model.stage=stage
    temp=path.with_suffix('.tmp')
    model.save(temp,metadata=extra)
    temp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',required=True);p.add_argument('--output',required=True)
    p.add_argument('--config',required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--seed',type=int,required=True);p.add_argument('--p',type=float)
    p.add_argument('--smoke-epochs',type=int);p.add_argument('--fit-limit',type=int)
    p.add_argument('--epoch-checkpoints',action='store_true')
    p.add_argument('--save-j-epochs',action='store_true',help='Keep each J epoch for paired selection studies')
    p.add_argument('--save-teacher-epochs',action='store_true',help='Keep each independent-teacher epoch without changing selection')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-after-epochs',type=int,help='Controlled interruption for resume verification')
    a=p.parse_args()
    start_implementation=implementation_hash()
    settings=json.loads(Path(a.config).read_text(encoding='utf-8'))
    method=settings['method'];variant=settings.get('variant','F111')
    if method!='OaPA':
        p.error('This trainer supports the OaPA method only')
    if a.resume:a.epoch_checkpoints=True
    if a.save_teacher_epochs:a.epoch_checkpoints=True
    if a.stop_after_epochs and (not a.epoch_checkpoints or a.stop_after_epochs<1):
        p.error('stop-after-epochs requires epoch-checkpoints and a positive count')
    recipe=copy.deepcopy(settings.get('recipe',RECIPE));model_settings={**DEFAULT_MODEL,**settings.get('model',{})}
    mpd_update=settings.get('mpd_update','weak_only')
    mpd_warmup_epochs=settings.get('mpd_warmup_epochs',0)
    if mpd_update not in ('weak_only','weak_first_then_all') or not isinstance(mpd_warmup_epochs,int) or mpd_warmup_epochs<0:
        p.error('Invalid MPD encoder update schedule')
    if mpd_update=='weak_first_then_all' and mpd_warmup_epochs<1:
        p.error('weak_first_then_all requires at least one warmup epoch')
    if a.smoke_epochs is not None:
        recipe['stages']={s:a.smoke_epochs for s in recipe['stages']}
    lr=settings.get('lr',1e-4);aux=settings.get('aux_scale',1.)
    if a.p is not None and not 0<a.p<1:p.error('P1 probability must be in (0,1)')
    outdir=Path(a.output)
    if a.resume and (not (outdir/'resume.pt').exists() or (outdir/'run.json').exists()):
        raise ValueError('Resume needs an unfinished run with resume.pt')
    if not a.resume and outdir.exists() and any(outdir.iterdir()):raise FileExistsError('Refusing to overwrite a run')
    outdir.mkdir(parents=True,exist_ok=True)
    if not a.resume:(outdir/'config.json').write_text(json.dumps(settings,indent=2))
    torch.set_num_threads(4);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    splits,dims,lengths,modalities,provenance=load_cache(a.cache)
    view_ce_weights=settings.get('view_ce_weights',{})
    if (not isinstance(view_ce_weights,dict) or
            any(name not in views(modalities) or not isinstance(weight,(int,float)) or
                not math.isfinite(weight) or weight<=0 for name,weight in view_ce_weights.items())):
        p.error('view_ce_weights must map legal views to positive finite weights')
    fit=splits['D_fit'];val=splits['D_val'];split_sizes={k:len(b) for k,b in splits.items()}
    for key in ('text_aux_ce_weight','all_views_aux_ce_weight','audio_expert_kd_weight'):
        if settings.get(key,0)!=0:
            p.error(f'{key} is not part of the OaPA release')
    if not fit.native.all() or not val.native.all():raise ValueError('Full-plan controlled cohort must be native complete')
    if a.fit_limit:
        fit=fit.select(slice(0,a.fit_limit));val=val.select(slice(0,a.fit_limit))
    del splits
    device=a.device
    model=build_oapa_backbone(dims,lengths,model_settings,variant).to(device)
    teachers=None
    distill=not (variant.startswith('F') and variant[1]=='0')
    if distill:
        teachers=nn.ModuleList([SequenceTeacher(d,model.cfg) for d in dims]).to(device)
    selection={};costs={};epochs=[];mask_schedule={};started=time.perf_counter();validation_calls=0
    recovery=None;elapsed_before=0.
    recovery_contract={'implementation':start_implementation,'config':settings,'recipe':recipe,
                       'seed':a.seed,'p':a.p,'fit_limit':a.fit_limit,'cache':file_sha(a.cache),
                       'save_j_epochs':a.save_j_epochs,'save_teacher_epochs':a.save_teacher_epochs}
    if a.resume:
        # Local trusted training state includes Python/NumPy RNG and optimizer.
        recovery=torch.load(outdir/'resume.pt',map_location=device,weights_only=False)
        if recovery['contract']!=recovery_contract:raise ValueError('Recovery contract changed')
        model.load_state_dict(recovery['model'],strict=True)
        if recovery.get('extra') is not None:model.load_extra_state(recovery['extra'])
        if teachers is not None:
            teachers.load_state_dict(recovery['teachers'],strict=True)
            if recovery['stage']!='teachers':teachers.requires_grad_(False).eval()
        selection=recovery['selection'];costs=recovery['costs'];epochs=recovery['epochs']
        mask_schedule=recovery['mask_schedule'];validation_calls=recovery['validation_calls']
        elapsed_before=recovery['elapsed']
        (outdir/'epochs.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in epochs))
    batch_size=recipe['batch_size']
    for sidx,(stage,budget) in enumerate(recipe['stages'].items()):
        if stage=='teachers' and teachers is None:continue
        if recovery is not None and sidx<recovery['stage_index']:continue
        tick=time.perf_counter();best=-1.;stale=0;updates=0;extra_forwards=0;stage_peak_before=0
        if stage=='teachers':
            target=teachers;target.requires_grad_(True)
        else:
            model.set_stage(stage);target=model
        parameters=[q for q in target.parameters() if q.requires_grad]
        stage_trainable=sum(q.numel() for q in parameters)
        if str(device).startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(device)
        if stage=='J':
            backbone={id(q) for q in model.backbone_parameters()}
            params=[{'params':[q for q in parameters if id(q) in backbone],'lr':lr*recipe['backbone_lr_scale']},
                    {'params':[q for q in parameters if id(q) not in backbone],'lr':lr}]
        else:params=parameters
        optimizer=getattr(torch.optim,recipe['optimizer'])(params,lr=lr,weight_decay=recipe['weight_decay'])
        scheduler=None
        rng=np.random.default_rng(a.seed+1000+sidx*1000)
        stage_masks=[];stage_orders=[]
        first_epoch=0
        if recovery is not None and sidx==recovery['stage_index']:
            optimizer.load_state_dict(recovery['optimizer'])
            best_temp=outdir/f'stage_{stage}.tmp'
            torch.save(recovery['best_checkpoint'],best_temp);best_temp.replace(outdir/f'stage_{stage}.pt')
            best=recovery['best'];stale=recovery['stale'];updates=recovery['updates']
            extra_forwards=recovery['extra_forwards'];stage_masks=recovery['stage_masks'];stage_orders=recovery['stage_orders']
            first_epoch=recovery['next_epoch'];tick-=recovery['stage_seconds']
            stage_peak_before=recovery.get('stage_peak_bytes',0)
            rng.bit_generator.state=recovery['numpy_generator']
            random.setstate(recovery['python_rng']);np.random.set_state(recovery['numpy_rng'])
            torch.set_rng_state(recovery['torch_rng'].cpu())
            if str(device).startswith('cuda'):torch.cuda.set_rng_state_all([r.cpu() for r in recovery['cuda_rng']])
            recovery=None
        for epoch in range(first_epoch,budget):
            target.train()
            # Independent, stage/epoch-keyed schedule is identical across
            # methods irrespective of extra teacher/auxiliary forwards.
            epoch_seed=a.seed+100000*(epoch+1)
            phase={'A':0,'B':1,'J':2,'fit':0,'student':0,'pretrain':0,'prompt':1,'teachers':3,'teacher':3}[stage]
            order_rng=np.random.default_rng(epoch_seed+phase*10000)
            order=order_rng.permutation(len(fit))
            mask_rng=np.random.default_rng(epoch_seed+phase*10000+77)
            masks=sample_masks(len(fit),len(dims),mask_rng,p=a.p)
            stage_masks.append(masks);stage_orders.append(order)
            per_epoch=[];scalar_terms={};coverage={k:0 for k in ('attempted','used','skipped','no_positive','no_negative')}
            for start in range(0,len(fit),batch_size):
                index=order[start:start+batch_size]
                batch=fit.select(torch.as_tensor(index)).to(device)
                mask=torch.as_tensor(masks[index],device=device)
                optimizer.zero_grad(set_to_none=True)
                stats={};teacher_logits=None
                if stage=='teachers':
                    losses=[F.cross_entropy(t(x),batch.labels) for t,x in zip(teachers,batch.features)]
                    loss=torch.stack(losses).mean();extra_forwards+=len(dims)
                else:
                    loss,stats,teacher_logits,out=backbone_loss(
                        model,batch,mask,teachers,stage,variant,aux,rng,
                        view_ce_weights,modalities,
                    )
                    if teacher_logits is not None:extra_forwards+=len(dims)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                logged_loss=float(loss.detach())
                weak_encoder_phase=(mpd_update=='weak_only' or (stage=='A' and epoch<mpd_warmup_epochs))
                if teacher_logits is not None and weak_encoder_phase:
                    with torch.no_grad():
                        probs=teacher_logits.softmax(-1).gather(2,batch.labels[:,None,None].expand(-1,len(dims),1)).squeeze(-1)
                        counts=mask.sum(0);scores=(probs*mask).sum(0)/counts.clamp_min(1);scores[counts==0]=torch.inf
                        coefficients=(counts>0)&(scores==scores.min())
                        for k,encoder in enumerate(model.projectors):
                            if not coefficients[k]:
                                for q in encoder.parameters():q.grad=None
                        stats['mpd_encoder_priority']=coefficients.cpu().tolist()
                elif teacher_logits is not None:
                    stats['mpd_encoder_priority']=[1]*len(dims)
                if not all(torch.isfinite(q.grad).all() for q in parameters if q.grad is not None):
                    raise FloatingPointError('Nonfinite gradient')
                if recipe.get('clip'):nn.utils.clip_grad_norm_(parameters,recipe['clip'])
                optimizer.step();updates+=1
                for k,v in stats.get('dcr_coverage',{}).items():coverage[k]+=v
                for k,v in stats.items():
                    if isinstance(v,(int,float)) and not isinstance(v,bool):scalar_terms.setdefault(k,[]).append(v)
                per_epoch.append(logged_loss)
            if hasattr(model,'epoch_end') and target is model:model.epoch_end()
            if stage=='teachers':
                # Independent unimodal teachers have no missing-view inference.
                teacher_scores=[]
                teachers.eval()
                with torch.inference_mode():
                    for k,t in enumerate(teachers):
                        predictions=[]
                        for start in range(0,len(val),batch_size):
                            predictions.extend(t(val.features[k][start:start+batch_size].to(device)).softmax(-1)[:,1].cpu().tolist())
                        teacher_scores.append(binary_metrics(val.labels.numpy(),predictions))
                validation={'selection_score':float(np.mean([r['macro_f1'] for r in teacher_scores])),'unimodal':teacher_scores}
            else:
                validation,_=evaluate(target,val,modalities,batch_size,device,p=a.p)
            validation_calls+=1;score=validation['selection_score']
            if scheduler:scheduler.step(score)
            if score>best:
                best=score;stale=0;selection[stage]={'epoch':epoch+1,'score':score,'tie':'earliest','split':'D_val'}
                if stage=='teachers':
                    temp=outdir/f'stage_{stage}.tmp'
                    torch.save({'teachers':teachers.state_dict()},temp);temp.replace(outdir/f'stage_{stage}.pt')
                else:save_checkpoint(outdir/f'stage_{stage}.pt',target,method,model_settings,variant,stage)
            else:stale+=1
            if a.save_teacher_epochs and stage=='teachers':
                temp=outdir/f'teachers_epoch_{epoch+1}.tmp'
                torch.save({'teachers':teachers.state_dict()},temp);temp.replace(outdir/f'teachers_epoch_{epoch+1}.pt')
            if a.save_j_epochs and stage=='J':
                save_checkpoint(outdir/f'J_epoch_{epoch+1}.pt',target,method,model_settings,variant,stage)
            counts={name:int((masks==np.array(bits)).all(1).sum()) for name,bits in views(modalities).items()}
            record={'stage':stage,'epoch':epoch+1,'mean_loss':float(np.mean(per_epoch)),
                    'mpd_encoder_update':('weak_only' if weak_encoder_phase else 'all') if stage in ('A','J') and distill else 'not_applicable',
                    'updates':updates,'selection_score':score,'validation':validation,'view_counts':counts,
                    'actual_drop_by_modality':(~masks).mean(0).tolist(),'dcr_coverage':coverage,
                    'training_terms':{k:float(np.mean(v)) for k,v in scalar_terms.items()},
                    'seconds_since_stage_start':time.perf_counter()-tick}
            epochs.append(record)
            with (outdir/'epochs.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps({k:v for k,v in record.items() if k not in ('validation',)}),flush=True)
            if a.epoch_checkpoints:
                state={'contract':recovery_contract,'stage':stage,'stage_index':sidx,'next_epoch':epoch+1,
                       'model':model.state_dict(),'extra':model.extra_state() if hasattr(model,'extra_state') else None,
                       'teachers':teachers.state_dict() if teachers is not None else None,
                       'optimizer':optimizer.state_dict(),'best':best,'stale':stale,'updates':updates,
                       'best_checkpoint':torch.load(outdir/f'stage_{stage}.pt',map_location='cpu',weights_only=True),
                       'extra_forwards':extra_forwards,'stage_masks':stage_masks,'stage_orders':stage_orders,
                       'selection':selection,'costs':costs,'epochs':epochs,'mask_schedule':mask_schedule,
                       'validation_calls':validation_calls,'elapsed':elapsed_before+time.perf_counter()-started,
                       'stage_seconds':time.perf_counter()-tick,'numpy_generator':rng.bit_generator.state,
                       'stage_peak_bytes':max(stage_peak_before,torch.cuda.max_memory_allocated(device)) if str(device).startswith('cuda') else 0,
                       'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
                       'torch_rng':torch.get_rng_state(),
                       'cuda_rng':torch.cuda.get_rng_state_all() if str(device).startswith('cuda') else []}
                temp=outdir/'resume.tmp'
                torch.save(state,temp);temp.replace(outdir/'resume.pt')
                if a.stop_after_epochs and len(epochs)>=a.stop_after_epochs:
                    print(json.dumps({'stopped_for_resume':str(outdir),'epochs':len(epochs)}),flush=True)
                    return
            if recipe.get('patience') and stale>=recipe['patience']:break
        checkpoint=torch.load(outdir/f'stage_{stage}.pt',map_location=device,weights_only=True)
        if stage=='teachers':teachers.load_state_dict(checkpoint['teachers']);teachers.requires_grad_(False).eval()
        else:
            target.load_state_dict(checkpoint['model'])
            if 'extra' in checkpoint:target.load_extra_state(checkpoint['extra'])
            if stage=='teacher':teacher_model.requires_grad_(False).eval()
        np.savez_compressed(outdir/f'mask_schedule_{stage}.npz',masks=np.stack(stage_masks),orders=np.stack(stage_orders))
        costs[stage]={'seconds':time.perf_counter()-tick,'updates':updates,'epochs':len(stage_masks),
                      'extra_forward_calls':extra_forwards,'includes_validation':True,
                      'trainable_parameters':stage_trainable,
                      'peak_allocated_bytes':max(stage_peak_before,torch.cuda.max_memory_allocated(device)) if str(device).startswith('cuda') else None}
        mask_schedule[stage]=file_sha(outdir/f'mask_schedule_{stage}.npz')
    model.set_stage('eval')
    validation,probabilities=evaluate(model,val,modalities,batch_size,device,p=a.p)
    save_checkpoint(outdir/'model.pt',model,method,model_settings,variant,getattr(model,'stage','fit'))
    np.savez_compressed(outdir/'D_val_predictions.npz',ids=np.array(val.ids),labels=val.labels.numpy(),
                        groups=np.array(val.groups),**probabilities)
    if implementation_hash()!=start_implementation:
        raise RuntimeError('Implementation changed during training; this attempt cannot be accepted')
    report={'status':'development_validation_selected' if a.seed==20261010 or a.smoke_epochs else 'trained_validation_selected_awaiting_frozen_test',
        'method':method,'variant':variant,'seed':a.seed,'protocol':'P0' if a.p is None else 'P1','p':a.p,
        'config':settings,'recipe':recipe,'model_settings':model_settings,'modalities':modalities,
        'input_dims':dims,'sequence_lengths':lengths,'cache_sha256':file_sha(a.cache),
        'configuration_sha256':file_sha(a.config),'implementation_sha256':start_implementation,
        'split_sizes':split_sizes,'optimized_splits':['D_fit'],'selection_split':'D_val',
        'D_test':'not_evaluated','selection':selection,'validation_calls':validation_calls,
        'D_val':validation,'mask_schedule_sha256':mask_schedule,'costs':costs,
        'seconds':elapsed_before+time.perf_counter()-started,'parameters':sum(q.numel() for q in model.parameters()),
        'model_sha256':file_sha(outdir/'model.pt'),'smoke_override':a.smoke_epochs,'fit_limit':a.fit_limit,
        'saved_j_epochs':a.save_j_epochs,
        'saved_teacher_epochs':a.save_teacher_epochs}
    (outdir/'run.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'complete':str(outdir),'D_test':'not_evaluated'}),flush=True)


if __name__=='__main__':main()
