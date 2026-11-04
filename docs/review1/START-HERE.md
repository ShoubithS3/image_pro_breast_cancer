# Review 1 on 5 November 2026

Your saved assignment plan lists a proposal and literature review worth 20%, with an 8 to 10 minute presentation and a report. Confirm the room, time, upload location and required personal details on the faculty/LMS notice. The saved plan reserves the 15 to 20 page report requirement for the final submission.

## What to use

- `MAMMO-LENS-Review-1.pptx`: 12 presentation slides and one reference slide. Speaker notes are embedded. Aim for 9 to 10 minutes.
- `Review-1-Proposal.pdf`: printable proposal with six primary references, methodology, dataset audit, measured baselines, limitations and timeline.
- `Review-1-Proposal.html`: readable report source. Open in a browser to read or print.
- `Speaking-Notes.md`: rehearse this alongside the slides. Explain the ideas in your own words.
- `Viva-and-Demo.md`: likely questions, concise answers and an optional two-minute demo.

Before submitting, add your name, registration number, team members and guide details in the format your faculty requires. No unverified personal details were invented in these files. Check the course title and institutional requirements.

## Tonight

1. Open the deck and rehearse once with a ten-minute timer.
2. Learn the problem, contribution boundary, patient split and three baseline numbers.
3. Read all six literature summaries and open the primary links, especially the GMIC paper.
4. Keep the deck, PDF and notes on your laptop and a backup drive.
5. If demonstrating the app, launch `START-WINDOWS.cmd` from the repository and confirm it opens before the review.

## The statement to remember

We have built the local research pipeline, audited 2,955 mammograms from 1,554 patients and completed three real handcrafted baselines. Our best preliminary image AUROC is 0.6249. Deep global/local and density-conditioned experiments are implemented but still require real training and evaluation.

Do not call AUROC 0.6249 an accuracy of 62.49%. Do not claim that texture patches are detected lesions, or that current results predict future cancer risk. Do not hide the weak subgroup and noise results: explain why they motivate the reliability study.

## Evidence and provenance

Aggregate numbers come from local real-data metrics and the manifest audit, checked again on 4 November 2026. `aggregate-results.json` contains only aggregate values. Source files remain under ignored `data/` and `artifacts/`. The code milestone used by the runs is `345a6be`. The previous local test result was 15 passing tests on 26 October. No new training was started to prepare this review.

The proposal reflects selected primary literature, not a completed systematic review. Deep-model and clinical performance claims remain pending.
