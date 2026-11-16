# Protected Link Materialization in the Legacy Archive

## Intent

Allow the four legacy SimNIBS 3.2.6 archive sources to materialize scientific
files referenced through symbolic links into the configured protected corrected
input dataset. The Shared archive must remain self-contained before the legacy
parscratch roots are retired.

The production audit found 5,600 links: 800 resolve inside the four archive
sources and 4,800 resolve inside the single configured protected materialization
root. It found no broken links and no unapproved external targets.

## Constraints

- Arbitrary external link targets remain rejected.
- Broken links remain rejected.
- Approved links are materialized as regular files by the existing
  `rsync --copy-links` archive operation.
- Protected materialization roots are not copied wholesale.
- Protected materialization roots remain outside the archive-source list and
  the retirement allowlist.
- A protected root that overlaps an archive or retirement target is rejected.
- The archive plan hash and preflight report bind the protected-root allowlist.
- Existing dataset finalization checkpoints remain reusable; no scientific
  source is modified.

## Approaches

1. **Allow all external links.** Rejected because it would silently expand the
   archive trust boundary and could materialize unrelated data.
2. **Preserve external links as links.** Rejected because the Shared archive
   would depend on parscratch paths that are being retired.
3. **Allow only configured protected materialization roots.** Selected because
   it matches the consolidation planner's existing protected-input policy while
   keeping the archive self-contained and fail-closed.

## Verification

Regression tests must prove that a link into a configured protected root is
materialized and checksum-verifies, while an arbitrary external link is still
rejected. The archive suite and complete consolidation suite must pass, and the
HPC launcher must retain valid shell syntax.
