# FairML — Project Study Guide (source)

The compiled output is `FairML_Project_Study_Guide.pdf` (28 pages).

**Compile:** `guide.tex` with pdfLaTeX. Works on Overleaf as-is (upload this
folder, set compiler to pdfLaTeX).

```
guide.tex             ← COMPILE THIS. Preamble, title page, Sections 1–2
part2_metrics.tex     ← Sections 3–4 (metrics, methodology)
part3_results.tex     ← Sections 5–7 (flowchart, results, paper preparation)
figures/              ← 9 result figures (vector PDF)
```

Sections 1 and 5 contain TikZ diagrams drawn inline — no image files needed.
Every number in this guide was read from the project's per-seed result CSVs or
computed directly from the code; the worked examples were verified against
`IndividualFairness.py`.
