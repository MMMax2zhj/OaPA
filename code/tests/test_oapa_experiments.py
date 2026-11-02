import sys
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np
import torch
from sklearn.metrics import f1_score
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from oapa.pipeline import OaPAVPA,calibration_rows,chosen_masks,route,score,FEATURES
from oapa.backbone import build_oapa_backbone
from oapa.objectives import DiscriminativeContrastiveRegularization
from oapa.protocol import views
from oapa.aggregation import VisibilityPredictionAggregator


def batch(m=3,n=31):
    return SimpleNamespace(ids=[f'fixture_{i}' for i in range(n)],split='D_val',
                           labels=torch.arange(n)%2,features=tuple(torch.ones(n,2,3)*(k+1) for k in range(m)),
                           __len__=lambda:n)


class FixtureBatch:
    def __init__(self,m=3,n=31):self.__dict__.update(batch(m,n).__dict__)
    def __len__(self):return len(self.ids)


class ConstantExpert:
    classes_=np.array([0,1])
    def predict_proba(self,x):
        return np.column_stack((np.full(len(x),.17),np.full(len(x),.83)))


class OaPAExperimentTests(unittest.TestCase):
    def test_train_cell_runs_the_main_pipeline_without_campaign(self):
        import run_oapa as core
        job=core.JOBS[0]
        with mock.patch.object(core,'fit_experts') as fit, \
             mock.patch.object(core,'train') as train, \
             mock.patch.object(core,'select') as select, \
             mock.patch.object(core,'calibrate') as calibrate:
            core.train_cell(job['id'])
        fit.assert_called_once_with([job['condition']])
        train.assert_called_once_with(job)
        select.assert_called_once_with(job)
        calibrate.assert_called_once_with(job)

    def test_test_cell_requires_a_known_run(self):
        import run_oapa as core
        with self.assertRaisesRegex(ValueError,'Unknown or duplicate run id'):
            core.test_cell('not-a-run')

    def test_protocol_manifest_is_complete_and_hash_locked(self):
        package=ROOT/'experiments/oapa'
        protocol=json.loads((package/'protocol.json').read_text())
        jobs=json.loads((package/'training_jobs.json').read_text())
        smoke=json.loads((package/'smoke_jobs.json').read_text())
        self.assertEqual(protocol['formal_training_jobs'],120)
        self.assertEqual(len(jobs),120)
        self.assertEqual(len({j['id'] for j in jobs}),120)
        self.assertEqual(len(smoke),protocol['smoke_training_jobs'])
        counts={name:sum(j['block']==name for j in jobs) for name in ('MAIN_P0','MAIN_P1','COMPONENTS','SEMANTIC')}
        self.assertEqual(counts,{'MAIN_P0':9,'MAIN_P1':63,'COMPONENTS':42,'SEMANTIC':6})
        for job in jobs:
            path=ROOT/job['config_path']
            self.assertTrue(path.is_file())
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),job['config_sha256'])
            config=json.loads(path.read_text())
            self.assertEqual(config['method'],'OaPA')
            self.assertEqual(config['variant'],job['variant'])
        for spec in smoke:
            argv=spec['argv']
            self.assertTrue((ROOT/argv[argv.index('--config')+1]).is_file())
            self.assertEqual(argv[0],'code/scripts/train_oapa.py')

    def test_component_variants_toggle_vpc(self):
        settings=dict(hidden=8,heads=2,layers=1,memory_layers=1,encoder_layers=1,
                      dropout=0,prompt_length=2,topics=2)
        disabled=build_oapa_backbone((8,6,4),(5,3,1),settings,'F000')
        enabled=build_oapa_backbone((8,6,4),(5,3,1),settings,'F010')
        self.assertFalse(disabled.cfg.use_completion)
        self.assertTrue(enabled.cfg.use_completion)
        self.assertEqual(disabled.variant,'F000')
        self.assertEqual(enabled.variant,'F010')

    def test_random_reference_dcr_keeps_semantic_contract(self):
        z=torch.randn(8,6)
        y=torch.tensor([0,0,0,0,1,1,1,1])
        ids=[f'id{i}' for i in range(8)]
        groups=[f'g{i}' for i in range(8)]
        semantic,_=DiscriminativeContrastiveRegularization()(z,y,ids,groups)
        random,coverage=DiscriminativeContrastiveRegularization(reference='random')(
            z,y,ids,groups,rng=np.random.default_rng(7)
        )
        self.assertTrue(torch.isfinite(semantic))
        self.assertTrue(torch.isfinite(random))
        self.assertEqual(coverage['used'],8)

    def test_p1_is_mixed_population_macro_f1(self):
        b=FixtureBatch();mods=('T','V','A');rng=np.random.default_rng(117)
        table=rng.random((len(b),7));chosen,_=chosen_masks(b,mods,.6)
        expected=[f1_score(b.labels.numpy(),table[np.arange(len(b)),c]>.5,labels=[0,1],average='macro',zero_division=0) for c in chosen]
        actual=score(table,b,mods,.6)
        np.testing.assert_allclose(actual['repetitions'],expected,atol=1e-15)
        per_view=np.array([f1_score(b.labels,table[:,i]>.5,labels=[0,1],average='macro') for i in range(7)])
        weighted=np.mean([np.bincount(c,minlength=7)@per_view/len(b) for c in chosen])
        self.assertGreater(abs(weighted-actual['mean_macro_f1']),1e-5)

    def test_p0_seven_view_average_and_ties(self):
        b=FixtureBatch();table=np.full((len(b),7),.5)
        result=score(table,b,('T','V','A'))
        expected=f1_score(b.labels,np.zeros(len(b)),labels=[0,1],average='macro')
        self.assertAlmostEqual(result['F_all'],expected)
        self.assertEqual(len(result['views']),7)

    def test_p0_selection_excludes_routed_singleton_views(self):
        b=FixtureBatch();names=list(views(('T','V','A')))
        table=np.zeros((len(b),len(names)))
        table[:,names.index('V')]=1
        table[:,names.index('A')]=1
        result=score(table,b,('T','V','A'),excluded_views=('A','V'))
        self.assertEqual(result['excluded_views'],['A','V'])
        self.assertNotIn('A',result['views'])
        self.assertNotIn('V',result['views'])
        self.assertEqual(result['selection_score'],result['F_all'])

    def test_singleton_experts_leave_joint_views_unchanged(self):
        b=FixtureBatch();original=np.full((len(b),7),.29);names=list(views(('T','V','A')))
        out=route(original,names,b,('T','V','A'),{'A':ConstantExpert(),'V':ConstantExpert()})
        for i,name in enumerate(names):
            np.testing.assert_array_equal(out[:,i],.83 if name in ('V','A') else original[:,i])
        np.testing.assert_array_equal(original,.29)

    def test_pf_routes_image_without_audio(self):
        b=FixtureBatch(2);names=list(views(('T','I')))
        out=route(np.full((len(b),3),.29),names,b,('T','I'),{'I':ConstantExpert()})
        np.testing.assert_array_equal(out[:,names.index('I')],.83)
        np.testing.assert_array_equal(out[:,names.index('TI')],.29)

    def test_matched_p_calibration_rows(self):
        b=FixtureBatch();mods=('T','V','A');extracted={}
        for i,(name,bits) in enumerate(views(mods).items()):
            extracted[name]={'fused':torch.full((len(b),2),float(i)),
                'branches':torch.full((len(b),3,2),float(i)),
                'mask':torch.tensor(bits)[None].expand(len(b),-1)}
        f,branches,masks,y=calibration_rows(extracted,b,mods,.4)
        chosen,official=chosen_masks(b,mods,.4)
        np.testing.assert_array_equal(f[:,0].numpy(),chosen.reshape(-1))
        np.testing.assert_array_equal(masks.numpy(),official.reshape(-1,3))
        torch.testing.assert_close(y,b.labels.repeat(10))

    def test_pipeline_vpa_matches_base_and_missing_branch_invariance(self):
        torch.manual_seed(10);base=VisibilityPredictionAggregator(3);gate=OaPAVPA(3)
        with torch.no_grad():base.scorer.weight.normal_();base.modality_bias.normal_()
        gate.load_state_dict(base.state_dict(),strict=False)
        fused=torch.randn(9,2);branches=torch.randn(9,3,2)
        mask=torch.tensor([[1,0,0],[0,1,0],[0,0,1],[1,1,1],[1,1,0],[1,0,1],[0,1,1],[1,0,0],[1,1,1]],dtype=torch.bool)
        for expected,actual in zip(base(fused,branches,mask),gate(fused,branches,mask)):
            torch.testing.assert_close(expected,actual,rtol=0,atol=0)
        other=branches.clone();other[~mask]=999
        torch.testing.assert_close(gate(fused,branches,mask)[0],gate(fused,other,mask)[0],rtol=0,atol=0)
        self.assertEqual(gate(fused,branches,mask)[1][:,1:][~mask].abs().sum().item(),0)

    def test_restricted_feature_masks(self):
        for kind,selected in FEATURES.items():
            gate=OaPAVPA(3,kind)
            self.assertEqual(gate.feature_selection.tolist(),[float(i in selected) for i in range(5)])
        gate=OaPAVPA(3,'mask')
        fused=torch.randn(5,2);branches=torch.randn(5,3,2);mask=torch.ones(5,3,dtype=torch.bool)
        gate(fused,branches,mask)[0].sum().backward()
        self.assertTrue((gate.scorer.weight.grad[0,[0,1,2,4]]==0).all())

    def test_six_table_schema_and_missing_row_rejection(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        from run_oapa import table_text,PACKAGE
        import json
        schema=json.loads((PACKAGE/'paper_output_schema.json').read_text())
        for name,spec in schema.items():
            if not isinstance(spec,dict):continue
            columns=spec['columns'];values={}
            if name in ('p0_sv_tt','p0_pf','p1'):
                for c in spec['datasets']:
                    for method in spec.get('rows_per_dataset',spec.get('rows')):
                        values[c+'|'+method]={v:None if (c,method)==('PF','MyGO_adapt') else {'mean':.4} for v in columns}
            else:values={r:{v:{'mean':.4} for v in columns} for r in spec['rows']}
            result=table_text(name,values,schema)
            self.assertIn('tab:'+name,result)
            if name in ('components','aggregation'):self.assertEqual(result.count('SV/$F_'),1 if name=='components' else 0)
            if name=='p0_pf' and 'MyGO_adapt' in spec.get('rows',[]):self.assertIn('MyGO (2026) & -- & -- & --',result)
            missing=dict(values);missing.pop(next(iter(missing)))
            with self.assertRaises((ValueError,KeyError)):table_text(name,missing,schema)

    def test_paired_event_bootstrap_uses_shared_groups(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        from run_oapa import bootstrap_difference
        y=np.array([0,1,0,1]);a=np.tile(y,(3,1)).astype(bool);b=~a
        result=bootstrap_difference(y,np.array(['one_event']*4),a,b,np.random.default_rng(20261031),32)
        self.assertEqual(result['groups'],1)
        self.assertEqual(result['observed_delta_pp'],100.)
        self.assertEqual(result['ci95_pp'],[100.,100.])

    def test_smoke_contract_has_required_calibration_fields(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        import run_oapa as core
        from oapa_preflight import smoke_specs
        specs=smoke_specs(core)
        self.assertEqual(len(specs),core.PROTOCOL['smoke_training_jobs'])
        for spec in specs:
            self.assertIn('block',spec)
            self.assertIn(spec['condition'],('SV','TT','PF'))
            if spec['method']=='OaPA' and spec['variant']=='F111' and spec['p'] is None:
                self.assertEqual(spec['block'],'MAIN_P0')


if __name__=='__main__':unittest.main()
