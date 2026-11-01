"""P0/P1-only reporting around the unchanged scientific implementation."""
import json
import numpy as np
from sklearn.metrics import f1_score


def install(core):
    original_evaluate = core.evaluate

    def evaluate(job):
        # The original evaluator uses this block solely to enable aggregation arms.
        # P0/P1 scoring itself uses p and the original checkpoint/gate identities.
        scoring_job = dict(job)
        if scoring_job['block'] == 'MAIN_P0':
            scoring_job['block'] = 'MAIN_P0_NO_AGGREGATION_ARMS'
        original_evaluate(scoring_job)

    def verify():
        checks = []
        for job in core.JOBS:
            folder = core.ROOT / job['run_dir']
            report = core.read(folder / 'paper_evaluation.json')
            core.verify_hashes(report['prediction_files'])
            with np.load(folder / 'paper_predictions.npz', allow_pickle=False) as raw:
                y, table, names = raw['labels'], raw['probabilities'], raw['views'].tolist()
                scores = [f1_score(y, table[:, i] > .5, labels=[0, 1], average='macro', zero_division=0)
                          for i in range(len(names))]
                np.testing.assert_allclose(scores, [report['result']['views'][v]['macro_f1'] for v in names], atol=1e-12, rtol=0)
                np.testing.assert_allclose(np.mean(scores), report['result']['F_all'], atol=1e-12, rtol=0)
                if job['p'] is not None:
                    mixed = [f1_score(y, table[np.arange(len(y)), chosen] > .5, labels=[0, 1], average='macro', zero_division=0)
                             for chosen in raw['chosen']]
                    assert len(mixed) == 10
                    np.testing.assert_allclose(mixed, report['result']['repetitions'], atol=1e-12, rtol=0)
                    np.testing.assert_allclose(np.mean(mixed), report['result']['mean_macro_f1'], atol=1e-12, rtol=0)
            checks.append({'run_id': job['id'], 'independent_sklearn_metric_match': True})
        core.write_json(core.OUT / 'independent_audit.json', {'passed': True, 'formal_jobs': len(checks), 'checks': checks,
                        'scope': 'P0/P1 only; component, aggregation and semantic controls excluded by user'})

    def aggregate():
        assert core.read(core.OUT / 'independent_audit.json')['passed']
        schema = {k: v for k, v in core.read(core.PACKAGE / 'paper_output_schema.json').items()
                  if k in ('p0_sv_tt', 'p0_pf', 'p1')}
        tables = {}
        def cell(jobs, metric):
            assert sorted(j['seed'] for j in jobs) == sorted(core.PROTOCOL['training_seeds'])
            values = [metric(core.read(core.ROOT / j['run_dir'] / 'paper_evaluation.json')['result'])
                      for j in sorted(jobs, key=lambda j: j['seed'])]
            return {'mean': float(np.mean(values)), 'seed_std': float(np.std(values, ddof=1)), 'seed_scores': values}
        for name, spec in schema.items():
            rows = {}
            for condition in spec['datasets']:
                jobs = [j for j in core.JOBS if j['condition'] == condition and
                        j['block'] == ('MAIN_P1' if name == 'p1' else 'MAIN_P0')]
                row = {}
                for column in spec['columns']:
                    if name == 'p1':
                        row[column] = cell([j for j in jobs if j['p'] == float(column)], lambda r: r['mean_macro_f1'])
                    else:
                        row[column] = cell(jobs, lambda r: r['views'][column]['macro_f1'])
                rows[condition + '|OaPA'] = row
            tables[name] = rows
        core.write_json(core.OUT / 'aggregate.json', {'tables': tables, 'schema': schema, 'formal_jobs': len(core.JOBS),
                        'three_seed_means': True, 'external_baselines_evaluated': False})

    def statistics():
        core.write_json(core.OUT / 'statistics.json', {'scope': 'P0/P1 OaPA only',
                        'seed_dispersion': 'sample SD and individual seed scores in aggregate.json',
                        'external_baselines_evaluated': False, 'control_comparisons': 'excluded by user scope reduction'})

    def export():
        report = core.read(core.OUT / 'aggregate.json')
        dest = core.OUT / 'export/tables'
        dest.mkdir(parents=True, exist_ok=True)
        for name, values in report['tables'].items():
            (dest / (name + '.tex')).write_text(core.table_text(name, values, report['schema']), encoding='utf-8')
        core.write_json(core.OUT / 'completion.json', {'status': 'COMPLETED', 'scope': 'P0/P1 only',
                        'formal_training_jobs': len(core.JOBS), 'full_gates': len(core.JOBS),
                        'experts': 5, 'export_files': core.fingerprint(dest.glob('*.tex')),
                        'actual_paper_modified': False, 'external_baselines_evaluated': False})

    core.evaluate, core.independent_verify = evaluate, verify
    core.aggregate, core.statistics, core.export = aggregate, statistics, export

