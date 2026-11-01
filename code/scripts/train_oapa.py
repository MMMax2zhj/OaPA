"""Train the OaPA backbone with staged VPC, MPD, and DCR objectives.

The independent evaluator is permitted only after a frozen selection manifest.
"""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
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


def backbone_loss(model,batch,mask,teachers,stage,aux_scale):
    out=model(batch.features,mask);m=mask.shape[1]
    ce=F.cross_entropy(out['logits'],batch.labels)
    loss=ce;stats={'ce':float(ce.detach())};teacher_logits=None
    if stage in ('A','J'):
        targets=batch.labels[:,None].expand(-1,m).flatten()
        branch=F.cross_entropy(out['branch_logits'].flatten(0,1),targets,reduction='none').reshape_as(mask)[mask].mean()
        loss=loss+.25*branch;stats['branch_ce']=float(branch.detach())
        with torch.no_grad():
            clean=tuple(torch.where(mask[:,k,None,None],x,torch.zeros_like(x)) for k,x in enumerate(batch.features))
            teacher_features=torch.stack([t.projector(x) for t,x in zip(teachers,clean)],1)
            teacher_logits=torch.stack([t.head(teacher_features[:,k]) for k,t in enumerate(teachers)],1)
        kd,feature=ModalityPrioritizedDistillation()(out['logits'],out['projected'],teacher_logits,teacher_features,mask)
        loss=loss+aux_scale*(.1*kd+.1*feature)
        stats.update(kd=float(kd.detach()),feature_kd=float(feature.detach()))
    if stage in ('B','J'):
        errors=[]
        for k,(pred,target) in enumerate(zip(out['reconstructed'],batch.features)):
            chosen=~mask[:,k]
            if chosen.any(): errors.append(1-F.cosine_similarity(pred[chosen],target[chosen].mean(1).detach(),dim=-1))
        reconstruction=torch.cat(errors).mean() if errors else ce*0
        loss=loss+aux_scale*.1*reconstruction
        stats['reconstruction']=float(reconstruction.detach())
        dcr,coverage=DiscriminativeContrastiveRegularization()(out['representation'],batch.labels,batch.ids,batch.groups)
        coverage.pop('pairs')
        loss=loss+aux_scale*.1*dcr
        stats.update(dcr=float(dcr.detach()),dcr_coverage=coverage)
    return loss,stats,teacher_logits,out


def save_checkpoint(path,model,stage):
    model.save(path,metadata={'stage':stage})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',required=True);p.add_argument('--output',required=True)
    p.add_argument('--config',required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--seed',type=int,required=True);p.add_argument('--p',type=float)
    p.add_argument('--smoke-epochs',type=int);p.add_argument('--fit-limit',type=int)
    a=p.parse_args()
    start_implementation=implementation_hash()
    settings=json.loads(Path(a.config).read_text(encoding='utf-8'))
    method=settings['method']
    if method != 'OaPA-Backbone': raise ValueError('Only OaPA-Backbone is supported')
    recipe=copy.deepcopy(settings.get('recipe',RECIPE));model_settings={**DEFAULT_MODEL,**settings.get('model',{})}
    if a.smoke_epochs is not None:
        recipe['stages']={s:a.smoke_epochs for s in recipe['stages']}
    lr=settings.get('lr',1e-4);aux=settings.get('aux_scale',1.)
    if a.p is not None and not 0<a.p<1:p.error('P1 probability must be in (0,1)')
    outdir=Path(a.output)
    if outdir.exists() and any(outdir.iterdir()):raise FileExistsError('Refusing to overwrite a run')
    outdir.mkdir(parents=True,exist_ok=True)
    (outdir/'config.json').write_text(json.dumps(settings,indent=2))
    torch.set_num_threads(4);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    splits,dims,lengths,modalities,provenance=load_cache(a.cache)
    fit=splits['D_fit'];val=splits['D_val'];split_sizes={k:len(b) for k,b in splits.items()}
    if not fit.native.all() or not val.native.all():raise ValueError('Full-plan controlled cohort must be native complete')
    if a.fit_limit:
        fit=fit.select(slice(0,a.fit_limit));val=val.select(slice(0,a.fit_limit))
    del splits
    device=a.device
    model=build_oapa_backbone(dims,lengths,model_settings).to(device)
    teachers=nn.ModuleList([SequenceTeacher(d,model.cfg) for d in dims]).to(device)
    selection={};costs={};epochs=[];mask_schedule={};started=time.perf_counter();validation_calls=0
    batch_size=recipe['batch_size']
    for sidx,(stage,budget) in enumerate(recipe['stages'].items()):
        if stage=='teachers' and teachers is None:continue
        tick=time.perf_counter();best=-1.;stale=0;updates=0;extra_forwards=0
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
        stage_masks=[];stage_orders=[]
        for epoch in range(budget):
            target.train()
            # Independent, stage/epoch-keyed schedule is identical across
            # methods irrespective of extra teacher/auxiliary forwards.
            epoch_seed=a.seed+100000*(epoch+1)
            phase={'A':0,'B':1,'J':2,'teachers':3}[stage]
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
                    loss,stats,teacher_logits,out=backbone_loss(model,batch,mask,teachers,stage,aux)
                    if teacher_logits is not None:extra_forwards+=len(dims)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                if teacher_logits is not None and stage=='A' and epoch<5:
                    with torch.no_grad():
                        probs=teacher_logits.softmax(-1).gather(2,batch.labels[:,None,None].expand(-1,len(dims),1)).squeeze(-1)
                        counts=mask.sum(0);scores=(probs*mask).sum(0)/counts.clamp_min(1);scores[counts==0]=torch.inf
                        coefficients=(counts>0)&(scores==scores.min())
                        for k,encoder in enumerate(model.projectors):
                            if not coefficients[k]:
                                for q in encoder.parameters():q.grad=None
                        stats['smp_coefficients']=coefficients.cpu().tolist()
                if not all(torch.isfinite(q.grad).all() for q in parameters if q.grad is not None):
                    raise FloatingPointError('Nonfinite gradient')
                optimizer.step();updates+=1
                for k,v in stats.get('dcr_coverage',{}).items():coverage[k]+=v
                for k,v in stats.items():
                    if isinstance(v,(int,float)) and not isinstance(v,bool):scalar_terms.setdefault(k,[]).append(v)
                per_epoch.append(float(loss.detach()))
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
            if score>best:
                best=score;stale=0;selection[stage]={'epoch':epoch+1,'score':score,'tie':'earliest','split':'D_val'}
                if stage=='teachers':torch.save({'teachers':teachers.state_dict()},outdir/f'stage_{stage}.pt')
                else:save_checkpoint(outdir/f'stage_{stage}.pt',target,stage)
            else:stale+=1
            counts={name:int((masks==np.array(bits)).all(1).sum()) for name,bits in views(modalities).items()}
            record={'stage':stage,'epoch':epoch+1,'mean_loss':float(np.mean(per_epoch)),
                    'updates':updates,'selection_score':score,'validation':validation,'view_counts':counts,
                    'actual_drop_by_modality':(~masks).mean(0).tolist(),'dcr_coverage':coverage,
                    'training_terms':{k:float(np.mean(v)) for k,v in scalar_terms.items()},
                    'seconds_since_stage_start':time.perf_counter()-tick}
            epochs.append(record)
            with (outdir/'epochs.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps({k:v for k,v in record.items() if k not in ('validation',)}),flush=True)
        checkpoint=torch.load(outdir/f'stage_{stage}.pt',map_location=device,weights_only=True)
        if stage=='teachers':teachers.load_state_dict(checkpoint['teachers']);teachers.requires_grad_(False).eval()
        else:
            target.load_state_dict(checkpoint['model'])
        np.savez_compressed(outdir/f'mask_schedule_{stage}.npz',masks=np.stack(stage_masks),orders=np.stack(stage_orders))
        costs[stage]={'seconds':time.perf_counter()-tick,'updates':updates,'epochs':len(stage_masks),
                      'extra_forward_calls':extra_forwards,'includes_validation':True,
                      'trainable_parameters':stage_trainable,
                      'peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else None}
        mask_schedule[stage]=file_sha(outdir/f'mask_schedule_{stage}.npz')
    model.set_stage('eval')
    validation,probabilities=evaluate(model,val,modalities,batch_size,device,p=a.p)
    save_checkpoint(outdir/'model.pt',model,'eval')
    np.savez_compressed(outdir/'D_val_predictions.npz',ids=np.array(val.ids),labels=val.labels.numpy(),
                        groups=np.array(val.groups),**probabilities)
    if implementation_hash()!=start_implementation:
        raise RuntimeError('Implementation changed during training; this attempt cannot be accepted')
    report={'status':'development_validation_selected' if a.seed==20261010 or a.smoke_epochs else 'trained_validation_selected_awaiting_frozen_test',
        'method':method,'seed':a.seed,'protocol':'P0' if a.p is None else 'P1','p':a.p,
        'config':settings,'recipe':recipe,'model_settings':model_settings,'modalities':modalities,
        'input_dims':dims,'sequence_lengths':lengths,'cache_sha256':file_sha(a.cache),
        'configuration_sha256':file_sha(a.config),'implementation_sha256':start_implementation,
        'split_sizes':split_sizes,'optimized_splits':['D_fit'],'selection_split':'D_val',
        'D_test':'not_evaluated','selection':selection,'validation_calls':validation_calls,
        'D_val':validation,'mask_schedule_sha256':mask_schedule,'costs':costs,
        'seconds':time.perf_counter()-started,'parameters':sum(q.numel() for q in model.parameters()),
        'model_sha256':file_sha(outdir/'model.pt'),'smoke_override':a.smoke_epochs,'fit_limit':a.fit_limit}
    (outdir/'run.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'complete':str(outdir),'D_test':'not_evaluated'}),flush=True)


if __name__=='__main__':main()
