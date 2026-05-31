# ACL Report TeX Project

This folder is the self-contained LaTeX project for the course report.

## Main Entry

Compile:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

On TeXPage or similar online LaTeX platforms, upload the whole `report/` folder and select `main.tex` as the main file.

## Files

- `main.tex`: English ACL-style report body.
- `custom.bib`: bibliography used by `main.tex`.
- `acl.sty`: ACL style file copied from the provided template.
- `acl_natbib.bst`: ACL bibliography style file copied from the provided template.
- `figures/`: report figures copied from `experiments/results/`.

## Notes

- The author block in `main.tex` is a placeholder and should be replaced before final submission.
- The Chinese Markdown counterpart is stored outside this TeX project at `docs/Report_ACL_zh.md`.
- The outline drafts were moved to `docs/report_outline.md` and `docs/report_outline_zh.md`.
- Report figures are generated once by the scripts under `scripts/report_assets/`; after regeneration, copy the needed PNG files from `experiments/results/` into `report/figures/`.
