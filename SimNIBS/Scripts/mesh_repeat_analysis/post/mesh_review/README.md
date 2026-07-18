# Mesh Render Review

This local web utility records accept, maybe, and decline decisions for
individual mesh-render images while maintaining subject-level inclusion state.
It uses only the Python standard library and does not modify the source images.

## Decision Rules

- Every image receives its own decision and optional note.
- Declining any image immediately makes the whole subject `declined` and removes
  all of that subject's remaining images from active review queues.
- A subject is `accepted` only when every active image discovered for that
  subject is accepted.
- A maybe decision moves the image from the new-image queue to the dedicated
  `Revisit maybe` queue.
- Adding an image for an accepted subject returns that subject to `in_review`
  until the new image is accepted.
- The interface opens an alert when the accepted-subject count first crosses
  the configured target, which defaults to 200.

The image directory defines the acceptance contract. If it contains ten repeat
directories, every render across all ten repeats must be accepted before the
subject is accepted. Point `--images` at a single repeat or curated render tree
when only that set should determine eligibility.

## Start Locally

From the repository's `SimNIBS/Scripts` directory:

```bash
python -m mesh_repeat_analysis.post.mesh_review.server \
  --images /path/to/render/images \
  --state-dir /path/to/mesh-review-state \
  --target 200
```

The image folder is scanned recursively. The browser opens at
`http://127.0.0.1:8765/`. Use `--port 0` to select any available port or
`--no-browser` to print the URL without opening it.

For the downloaded Left Hippocampus Data 01 renders, an example is:

```bash
python -m mesh_repeat_analysis.post.mesh_review.server \
  --images /home/boyan/sandbox/Jake_Data/mesh-wall/Left_Hippocampus_tissue_front_back_orthographic_all_repeats/Left_Hippocampus_Data_01/renders \
  --state-dir /home/boyan/sandbox/Jake_Data/mesh-review/left-hippocampus-data-01 \
  --target 200
```

Use the same `--state-dir` to resume. The server rescans the image directory at
startup while retaining decisions for paths already in the database.

## Supported Naming

By default, subject IDs are extracted with `sub-[A-Za-z0-9]+`. Existing mesh-QC
paths are understood directly:

```text
renders/tissues_back/tag_07_compact_bone/
00001__sub-CC120001__unknown_repeat__mesh.png
```

Flat names such as this are also supported:

```text
sub-CC120001__tag_07_compact_bone__back.png
```

Override unusual datasets with `--subject-regex` or `--tissue-regex`. The first
capturing group is used when the expression contains one.

## Review Workflow

1. Keep `Queue: New images` selected for the initial pass.
2. Choose subject-first ordering to inspect all tissues for one subject together,
   or tissue-first ordering to compare the same tissue across subjects.
3. Click Accept, Maybe, or Decline subject. Keyboard shortcuts `A`, `M`, and `D`
   are available when the note field is not focused.
4. Use `Revisit maybe` for deferred images.
5. Use the Subjects view to inspect aggregate state and search for a subject ID.
6. Click Rescan folder after adding new renders.
7. Click Export when the cohort is ready.

Undo restores the immediately preceding decision. If that decision declined a
subject, undo makes its remaining images eligible again.

## Persistent State And Exports

The state directory contains:

```text
mesh_review.sqlite3
exports/accepted_subjects.txt
exports/maybe_subjects.txt
exports/declined_subjects.txt
exports/subject_summary.csv
exports/image_decisions.csv
exports/review_manifest.json
```

SQLite is the authoritative state and is updated on every decision. The three
subject-ID lists and `subject_summary.csv` are also refreshed after every
decision. `image_decisions.csv` is refreshed by Export, Rescan, and clean server
shutdown to avoid rewriting the full image table after every click.

Stop the server before copying or archiving the state directory. Backing up the
whole directory preserves the database and all human-readable exports.

## Running Remotely

The default bind address is local-only. If the images and state must remain on
an HPC node, start with `--no-browser` and create an SSH tunnel from the local
machine:

```bash
ssh -L 8765:127.0.0.1:8765 user@login-host
```

Then open `http://127.0.0.1:8765/` locally. Do not bind the review server to a
public interface unless access is protected separately.
