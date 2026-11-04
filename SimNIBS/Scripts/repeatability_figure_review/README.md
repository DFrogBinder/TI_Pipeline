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

