# Repository Cleanup Guide

This document describes how to bring the Skip-Vision repository to a clean,
release-quality state. Apply the changes on a fresh branch, and keep a copy of
any large artifacts you still need outside Git.

## What To Remove

The current repository contains three categories of files that should never be
tracked in a public code repository:

1. **Binary training artifacts** under `checkpoints/` - model weights,
   optimizer states, and prediction JSONL files. These are the single largest
   source of repository bloat.
2. **Machine-generated caches** - `__pycache__/`, `.pyc`, `.DS_Store`,
   and stray `.nfs*` files inside the package.
3. **One-off experiment variants** - dozens of `llava_arch_*` and
   `llama_squeeze_expand*` files that encode temporary ablations rather than
   the released model. The released model only needs
   `llava_arch_index_q_tome.py` plus its LLaMA wrapper.

## Exact Cleanup Commands

Run these from the repository root on a new branch. They remove files from
Git tracking only; add the provided `.gitignore` first so nothing is
accidentally re-added.

```bash
git checkout -b cleanup/repo-hygiene
git add .gitignore
git commit -m "Add .gitignore for caches, data, and training artifacts"

# 1. Remove all committed caches and OS junk.
git rm -r --cached $(git ls-files | grep -E '(__pycache__|\.pyc$|\.DS_Store$|\.nfs)')

# 2. Stop tracking checkpoints and raw data. Keep the files on disk.
git rm -r --cached checkpoints playground debug

# 3. Archive the one-off experiment variants, then delete them from the index.
mkdir -p /tmp/skip-vision-ablations
git ls-files 'llava/model/language_model/llama_squeeze_expand*' \
  | xargs -I{} cp --parents {} /tmp/skip-vision-ablations/
git ls-files 'llava/model/llava_arch_hd*' 'llava/model/llava_arch_vis_supervision.py' \
  | xargs -I{} cp --parents {} /tmp/skip-vision-ablations/
git rm --cached \
  'llava/model/language_model/llama_squeeze_expand*' \
  'llava/model/llava_arch_hd*' \
  'llava/model/llava_arch_vis_supervision.py'

git commit -m "Remove caches, checkpoints, and unreleased experiment variants"
```

## Shrinking Git History

Removing files in a new commit stops future bloat but does not shrink clones,
because the blobs are still reachable in older commits. To shrink history:

1. Ensure every large weight is backed up outside the repository.
2. Install `git-filter-repo`.
3. Force-push a rewritten history and ask collaborators to re-clone.

```bash
pip install git-filter-repo
git filter-repo --invert-paths \
  --path checkpoints/ \
  --path debug/ \
  --path-glob '*__pycache__*' \
  --path-glob '*.pyc' \
  --path-glob '*.DS_Store' \
  --path-glob '*llama_squeeze_expand*' \
  --path-glob 'llava/model/llava_arch_hd*'
git push --force origin main
```

## Recommended Final Layout

```text
.gitignore
LICENSE
README.md
requirements.txt
docs/               User-facing documentation only
llava/              Released model, training, and eval code
recipe/             The four clean scripts in this kit
scripts/            DeepSpeed configs and dataset utilities
index.html          Project page
images/             Paper figures referenced by the README
static/             Assets for the project page
```

Folders such as `playground/` and `checkpoints/` should be created at runtime
by the recipes, but never contain committed content.
