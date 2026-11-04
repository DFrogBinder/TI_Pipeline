# Repeatability figure review

Build a source-bound figure-review companion without changing either the
downloaded combined analysis or the manuscript:

```bash
python -m repeatability_figure_review \
  --source-root /path/to/final_132_repeatability_combined_analysis_v1 \
  --manuscript-root /path/to/Manuscript \
  --output-root /path/to/final_132_repeatability_combined_analysis_v1_figure_review_v1
```

The production bootstrap uses 20,000 resamples at each run count from 2 to 40.
The PDF renderer uses the system ReportLab installation through `/usr/bin/python3`.

Build the separate image-only clarity handoff with:

```bash
python -m repeatability_figure_review.paper_figures \
  --source-root /path/to/final_132_repeatability_combined_analysis_v1 \
  --manuscript-root /path/to/Manuscript \
  --output-root /path/to/final_132_repeatability_paper_figures_v2_clarity
```

This command creates exactly 12 PNG files in three folders. Ten figures are
regenerated from the canonical analysis tables with reader-facing terminology;
the two legacy structural figures are copied byte-for-byte because their source
metric tables are unavailable in the downloaded analysis. Existing output roots
are rejected, and the analysis and manuscript trees are verified unchanged.
