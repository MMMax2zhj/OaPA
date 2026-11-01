"""OaPA expert routing, checkpoint selection, and VPA calibration."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from .aggregation import VPA_FEATURES, VisibilityPredictionAggregator
from .backbone import OaPABackbone
from .protocol import binary_metrics, fixed_masks, views

FEATURES = VPA_FEATURES


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(path)


class OaPAVPA(VisibilityPredictionAggregator):
    def __init__(self,m,kind='full'):
        super().__init__(m,kind=kind)


def restore(path,report,device):
    model=OaPABackbone.load(path,device).eval()
    if tuple(model.cfg.input_dims)!=tuple(report['input_dims']):
        raise ValueError('Checkpoint input dimensions do not match the report')
    if tuple(model.cfg.sequence_lengths)!=tuple(report['sequence_lengths']):
        raise ValueError('Checkpoint sequence lengths do not match the report')
    if model.variant!=report['variant']:
        raise ValueError('Checkpoint variant does not match the report')
    return model


def extract(model,batch,modalities,device,batch_size=32):
    result={}
    model.eval()
    for name,bits in views(modalities).items():
        fused=[];branches=[]
        with torch.inference_mode():
            for start in range(0,len(batch),batch_size):
                part=batch.select(slice(start,start+batch_size)).to(device)
                mask=torch.tensor(bits,device=device,dtype=torch.bool)[None].expand(len(part),-1)
                output=model(part.features,mask)
                fused.append(output['logits'].cpu())
                if 'branch_logits' in output:branches.append(output['branch_logits'].cpu())
        result[name]={'fused':torch.cat(fused),
                      'branches':torch.cat(branches) if branches else None,
                      'mask':torch.tensor(bits,dtype=torch.bool)[None].expand(len(batch),-1).clone()}
    return result


def chosen_masks(batch,modalities,p):
    bits=np.asarray(list(views(modalities).values()),dtype=bool)
    masks=np.stack([fixed_masks(batch.ids,len(modalities),batch.split,p,r) for r in range(10)])
    matches=(masks[:,:,None,:]==bits[None,None,:,:]).all(-1)
    if not (matches.sum(-1)==1).all():raise ValueError('Invalid masks')
    return matches.argmax(-1),masks


def score(table,batch,modalities,p=None):
    table=np.asarray(table)
    if table.shape!=(len(batch),len(views(modalities))) or not np.isfinite(table).all():
        raise ValueError('Prediction table mismatch')
    y=batch.labels.cpu().numpy()
    per_view={v:binary_metrics(y,table[:,i]) for i,v in enumerate(views(modalities))}
    values=[x['macro_f1'] for x in per_view.values()]
    result={'views':per_view,'F_all':float(np.mean(values)),'F_worst':float(min(values)),
            'F_full':per_view[''.join(modalities)]['macro_f1']}
    if p is not None:
        chosen,masks=chosen_masks(batch,modalities,p)
        repetitions=[binary_metrics(y,table[np.arange(len(batch)),r])['macro_f1'] for r in chosen]
        result.update(mean_macro_f1=float(np.mean(repetitions)),mask_std=float(np.std(repetitions,ddof=1)),
                      repetitions=repetitions,mask_sha256=hashlib.sha256(masks.tobytes()).hexdigest())
    return result


def neural_table(extracted,gate=None,alpha=None):
    table=[]
    with torch.inference_mode():
        for item in extracted.values():
            fused=item['fused'].softmax(-1)[:,1]
            if gate is not None:probability=gate(item['fused'],item['branches'],item['mask'])[0]
            elif alpha is not None:
                branch=item['branches'].softmax(-1)[:,:,1]
                visible=(branch*item['mask']).sum(1)/item['mask'].sum(1)
                probability=(1-alpha)*fused+alpha*visible
            else:probability=fused
            table.append(probability.numpy())
    return np.stack(table,1)


def route(table,names,batch,modalities,experts):
    table=np.asarray(table).copy()
    for view,expert in experts.items():
        if view not in names:continue
        k=modalities.index(view)
        # Called only for the corresponding singleton column. No hidden feature is used.
        x=batch.features[k].mean(1).cpu().numpy()
        classes=list(expert.classes_)
        table[:,names.index(view)]=expert.predict_proba(x)[:,classes.index(1)]
    return table


def calibration_rows(extracted,batch,modalities,p):
    values=list(extracted.values())
    if p is None:
        return tuple(torch.cat([x[key] for x in values]) for key in ('fused','branches','mask'))+(batch.labels.repeat(len(values)),)
    chosen,_=chosen_masks(batch,modalities,p)
    chosen=torch.as_tensor(chosen.reshape(-1))
    index=torch.arange(len(batch)).repeat(10)
    return tuple(torch.stack([x[key] for x in values],1)[index,chosen] for key in ('fused','branches','mask'))+(batch.labels.repeat(10),)


def fit_gate(extracted,batch,modalities,p=None,kind='full',steps=200):
    torch.manual_seed(31337)
    gate=OaPAVPA(len(modalities),kind)
    fused,branches,masks,targets=calibration_rows(extracted,batch,modalities,p)
    frequency=torch.bincount(targets,minlength=2).float()
    if not (frequency>0).all():raise ValueError('Calibration requires both classes')
    sample_weight=.5*len(targets)/frequency[targets]
    optimizer=torch.optim.AdamW(gate.parameters(),lr=.03,weight_decay=.02)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        probability,weights=gate(fused,branches,masks)
        loss=F.binary_cross_entropy(probability.clamp(1e-6,1-1e-6),targets.float(),weight=sample_weight)
        loss=loss+.2*weights[:,1:].sum(1).mean()
        if not torch.isfinite(loss):raise FloatingPointError('Gate loss')
        loss.backward();optimizer.step()
    return gate.eval(),{'kind':kind,'steps':steps,'branch_penalty':.2,'balanced_bce':True,
                        'lr':.03,'weight_decay':.02,'seed':31337,'p':p,'split':batch.split,
                        'rows':len(targets),'loss':float(loss.detach())}
