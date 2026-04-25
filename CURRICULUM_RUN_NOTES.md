# After-Curriculum-Run Checklist

After you run `LegaLoom_FullCurriculum.ipynb` on Colab and download the four artifacts (`reward_curves.png`, `before_after.png`, `training_scores.json`, `training_log.json`):

## 1. Drop artifacts into repo root

Replace the existing four files in the repo root. The notebook also generates them with the same filenames, so this is a straight overwrite.

```
reward_curves.png
before_after.png
training_scores.json
training_log.json
```

## 2. Decide: are the new numbers better?

Open `training_scores.json`. Compare to the old numbers:

| Task | Old baseline | Old trained | New baseline | New trained |
|------|---:|---:|---:|---:|
| easy   | 0.186 | 0.324 | _check_ | _check_ |
| medium | 0.450 | 0.336 | _check_ | _check_ |
| hard   | 0.078 | 0.126 | _check_ | _check_ |
| expert | 0.200 | 0.316 | _check_ | _check_ |

**If `task_medium` no longer regresses** (i.e. new trained ≥ new baseline), the curriculum worked. Push.

**If results are uniformly better**, even better. Push.

**If results got worse**, revert: `git restore reward_curves.png before_after.png training_scores.json training_log.json` and keep the proven 40-step results.

## 3. Update the README Results section

Cell 9 of the notebook prints a markdown table with the new numbers. Copy that block and paste it into `README.md`, replacing the existing table that starts with `| Task | Baseline | After GRPO | Δ |`.

Also update three things around the table:

- **Setup paragraph** (currently says "40 GRPO steps total — 20 on `task_easy` then 20 on `task_hard`"): change to "**80 GRPO steps total** — 20 each on `task_easy`, `task_medium`, `task_hard`, `task_expert` (full curriculum). 10 fresh-seed episodes per task in eval (was 5)."
- **Notebook reference**: change `LegaLoom_QuickTrain.ipynb` to `LegaLoom_FullCurriculum.ipynb` in the Setup paragraph.
- **"Why task_medium regressed" subsection**: if medium no longer regresses, **delete the entire subsection** — it's no longer true. If medium still regresses (less likely with curriculum training), keep it and update the numbers.

## 4. Update the blog post

Same table update in `blog_post.md`. Same paragraph adjustments around it.

## 5. Update the demo script

Open `demo_script.md`. Update the `task_hard` baseline/trained numbers (currently 0.078 / 0.126) and the `task_medium` regression mention if no longer applicable.

## 6. Commit and push

```bash
git add -A
git commit -m "Curriculum training: 80 steps across all 4 tasks, 10-episode eval"
git push
```

The HF Space picks up the push automatically via the Actions workflow.

## 7. Re-record the video

The before/after numbers in your video script should match the new results. If `task_hard` improved more, use that. If `task_medium` no longer regresses, the demo gets cleaner — drop the "honest about trade-off" segment or shorten it.

---

**If anything fails during the run** (out of memory, timeout, module import error), the existing 40-step artifacts and `LegaLoom_QuickTrain.ipynb` remain untouched in the repo. You can always fall back to those.
