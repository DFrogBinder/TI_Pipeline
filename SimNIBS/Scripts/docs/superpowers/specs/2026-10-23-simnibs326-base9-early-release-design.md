# SimNIBS 3.2.6 Base-Nine Early Release Design

## Intent

Start the corrected-label v3 FEM work for the nine successfully meshed subjects
while `sub-CC520083` is investigated, without weakening the full ten-subject
provenance gates or mixing partial and final campaign artifacts.

## Scope

- Preserve `full10` as the default cohort and preserve every existing path,
  module, resource, stimulation, retry, and validation setting.
- Add an opt-in `base9` cohort containing the original balanced-ten subjects
  except `sub-CC520083`.
- Collect, render, review, and promote the nine existing head models under a
  cohort-specific immutable manifest.
- Run 9 subjects x 2 targets x 2 conditions x 40 repeats: 1,440 simulation
  tasks and 2,880 component FEM solves. The first release is the 720 remesh
  tasks; fixed-mesh tasks remain gated on remesh completion and median-mesh
  selection.
- Add a one-subject diagnostic head-model submission for `sub-CC520083`. It
  performs no FEM solve, retries zero times, cleans only known generated
  headreco artifacts, and retains an isolated attempt log and Gmsh working
  directory.

## Architecture

`SIMNIBS326H_COHORT` selects `full10` or `base9`; `submit.sh` exposes this as
`--cohort=base9` and exports it to every submitted job. Python resolves the
active subject tuple, task counts, validation thresholds, and default result
roots from that profile. Full-ten head-model collection remains under
`_pipeline/headmodels`; base-nine collection uses
`_pipeline/headmodels/base9`, including independent QC panels, checksums,
manifest, and promotion receipt.

Base-nine result roots are separate from both the protected completed studies
and the pending full-ten v2 roots. The existing full ten-subject reference-stage
receipt is reused because it already proves the staged references for all nine
base subjects. Simulation tasks remain bound to the exact active-cohort
promotion hash.

The diagnostic submission reuses the existing head-model worker with a
single-element array and `SIMNIBS326H_MAX_RETRIES=0`. Every head-model attempt
runs in a job-specific working directory, preventing concurrent jobs from
sharing `gmsh_log.txt`. Retry cleanup removes only generated STL/OFF/MSH,
tissue NIfTI, and GEO files from the staged subject root; coordinate references
and provenance receipts remain untouched.

## Safety and non-goals

- No existing head model, completed v3/v4 result, corrected label, or reference
  input is overwritten.
- No automatic promotion occurs; human review and the exact manifest hash are
  still required.
- No fixed-mesh task is submitted before the remesh completion and ROI gates.
- This change does not choose whether `sub-CC520083` is repaired or replaced,
  and it does not combine the eventual tenth subject with the base-nine results.
- The final 9+1 aggregation will be a separate, provenance-bound analysis step.

## Alternatives considered

Waiting for ten subjects preserves the original workflow but loses compute
time. Bypassing promotion would be faster but would invalidate the audit trail.
An isolated base-nine release was selected because its nine subject outputs
remain valid under either later outcome for the tenth subject.

## Verification

- Existing full-ten tests and launcher behavior remain unchanged by default.
- New tests prove exact base-nine subject membership, roots, manifests,
  promotion binding, `0-359%50` simulation arrays, `0-8%9` ROI arrays, and
  360/720 completion thresholds.
- Diagnostic launcher tests prove a one-element array, zero retries, no
  collector, and isolated work/log paths.
- Cleanup tests prove removal of the observed failed-attempt artifacts while
  preserving the staged reference allowlist.
- Run focused pytest, complete package pytest, `bash -n`, and `git diff --check`
  before commit and push.
