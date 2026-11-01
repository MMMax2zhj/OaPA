# OaPA 4R Experiment Protocol

**CoMPA4R is the complete code implementation of OaPA.** The paper-facing
method name is OaPA; CoMPA4R refers to this complete four-route codebase.
This directory freezes that implementation:

- staged MPD backbone training;
- VPC and DCR component switches;
- D_fit-only audio LDA and visual/image RBF experts;
- deployed Stage-J checkpoint selection before VPA;
- visibility-constrained VPA calibration;
- frozen test evaluation, independent audit, and table export.

`protocol.json` defines the full 120-job matrix and the 72-job paper scope.
`training_jobs.json` enumerates the full matrix. `smoke_jobs.json` contains the
M0-only shortened runs. Feature caches are not included.

The complete pipeline is:

```bash
python code/scripts/oapa4r_env.py --output outputs/oapa4r/fresh_agent_environment_witness.json
python code/scripts/oapa4r.py --phase preflight --execute
python code/scripts/oapa4r.py --phase campaign --execute
```

`code/scripts/oapa4r_main.py` runs the paper-scoped `MAIN_P0` and `MAIN_P1`
blocks after M0 acceptance. See `protocol.json` for evidence paths,
calibration splits, checkpoint selection rules, and export schemas.
