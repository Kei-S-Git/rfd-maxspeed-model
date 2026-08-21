# CLAUDE.md — rfd-maxspeed-model

**This repository is public.** It is the code and data availability package cited by
a manuscript that is currently under editorial review.

## Do not change it while the manuscript is under review

Anything committed here is visible to editors and reviewers who follow the data
availability link. Treat every edit as a public statement about the manuscript.

- **Keep the README venue-neutral.** Do not name a journal in it. The manuscript's
  venue has changed during preparation, and a README that names one goes stale
  silently — it once stated a submission that had not happened, which invites the
  worst available reading.
- **Do not change `src/figstyle.py`'s dimensions.** Those constants produced the
  figures that were actually submitted.
- **Do not add internal research-management notes here** — submission identifiers,
  editorial correspondence, review history. Those belong in the private repositories.

## Where the work happens

This is a publication artifact, not a workspace. Analysis, drafting, and discussion
for this manuscript happen in the private sister repository `../sprint-dynamics-third`.

## About the contents

- **No new measurements were made.** Every empirical value is transcribed from a
  published table, and appears in the source next to its citation, its sample size,
  and its role (calibration or test).
- The paper's distinct contribution is an algebraic audit showing that the
  calibration removes the rate parameter, plus an out-of-sample test across four
  laboratories. See `../portfolio/STANDING_CHECKS.md` §1, which is written from
  this case.
