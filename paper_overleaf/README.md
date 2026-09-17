# Mapping the Group–Individual Fairness Trade-off — Overleaf Project

IEEE conference-format paper (`IEEEtran`, `conference` option), fully
self-contained. Upload the ZIP to Overleaf and compile — nothing else to add.

## Quick start on Overleaf

1. **New Project → Upload Project** → select the ZIP.
2. Open **`main.tex`** — this is the file to compile.
3. Menu → **Compiler: pdfLaTeX** (this is Overleaf's default).
4. Click **Recompile**.

The bibliography runs through BibTeX and resolves automatically on Overleaf.
If citations show as `[?]` on the very first build, hit Recompile once more —
that is the normal LaTeX → BibTeX → LaTeX → LaTeX sequence settling.

## Folder structure

```
.
├── main.tex                  ← COMPILE THIS. Preamble, title, abstract, \input order
├── refs.bib                  ← bibliography (36 entries, IEEE style)
├── IEEEtran.cls              ← IEEE class file (vendored so the project is self-contained)
├── IEEEtran.bst              ← IEEE bibliography style (vendored)
├── paper_preview.pdf         ← the compiled output, for reference
├── README.md                 ← this file
├── sections/
│   ├── 01_introduction.tex
│   ├── 02_related_work.tex
│   ├── 03_method.tex         ← equations (1)–(4), Algorithm 1, TikZ pipeline (Fig. 1)
│   ├── 04_experimental_setup.tex   ← Table I (datasets)
│   ├── 05_results.tex        ← Figs. 2–10, Tables II–IV
│   ├── 06_discussion.tex
│   ├── 07_limitations.tex
│   └── 08_conclusion.tex
└── figures/                  ← all 9 figures, vector PDF, IEEE column widths
    ├── surrogate_validation.pdf     Fig. 2   surrogate vs. hard metric
    ├── double_dissociation.pdf      Fig. 3   the core result
    ├── beta_cross_effect.pdf        Fig. 4   DPD & GE(2) vs. β
    ├── dpd_vs_ge_tradeoff.pdf       Fig. 5   trade-off frontier
    ├── pareto_frontiers.pdf         Fig. 6   accuracy vs. DPD + baselines
    ├── fair_vs_baseline.pdf         Fig. 7   selected model vs. baseline
    ├── selection_diagnostic.pdf     Fig. 8   val vs. test DPD (German diagnosis)
    ├── heatmap_dpd.pdf              Fig. 9   α×β surface, DPD
    └── heatmap_ge.pdf               Fig. 10  α×β surface, GE(2)
```

Figure 1 (the pipeline diagram) is drawn inline in TikZ inside
`sections/03_method.tex` — it has no image file, so there is nothing to break.

## Verification status

Compiled locally with Tectonic 0.17 (TeX Live package set) before delivery:

| Check | Result |
|---|---|
| Compilation errors | **0** |
| Undefined references | **0** |
| Undefined citations | **0** |
| Missing figure files | **0** |
| Overfull `\hbox` warnings | **0** |
| Output | 11 pages |

All 36 bibliography entries are cited; no orphan entries. Every file in
`figures/` is referenced by exactly one `\includegraphics`, and every
`\includegraphics` target exists.

## Notes before you submit

- **Page count.** The paper is 11 pages including references. Most IEEE
  conferences cap the main text at 6 or 8 pages. To trim, the two heatmap
  figures (Figs. 9–10) and their subsection are the natural first cut —
  they are supplementary — followed by Table IV, whose content is stated in
  the text of Section V-F.
- **Author block.** Name, department, and affiliation in `main.tex` were
  filled in from the repository's git identity and email domain. Please
  confirm the department name is correct before submitting.
- **Metric naming.** What the code stores as `Theil Index` is AIF360's
  generalized entropy index with α = 2. The Theil index proper is α = 1, so
  the paper deliberately labels this quantity **GE(2)** throughout. This is a
  naming correction relative to the codebase, not a change of numbers.
- **Relative paths.** `\graphicspath{{figures/}}` is set in `main.tex` and all
  includes use bare filenames, so the project relocates cleanly. No filename
  contains a space or a special character.

## Reproducing the numbers

Every figure and table is generated from the per-seed CSVs produced by the
experiment pipeline in the parent repository:

```bash
python new_experiment/run_experiment.py --dataset all --seeds 0-9
python new_experiment/analysis/run_analysis.py --dataset all
```
