# Layer A failure gallery

26 of 50 queries missed at rank 1; the 20 most useful are shown in `gallery.jpg`.
Images are local only (copyrighted source frames) and are ignored by git.

| # | Query | Type | Predicted | Ground truth | Cause |
|---|---|---|---|---|---|
| q002 | where he explains what the introduction needs to do | speech | 00:133 | 02:21–02:30 | Speech layer missing; the top hit was the “Essay Introductions” slide 8 s early — a near miss. |
| q003 | where he says lack of support at home | speech | 19:34 | 11:46–12:15 | Speech layer missing; matched the end-card playbook cover instead. |
| q005 | where he explains the passive voice | speech | 00:11 | 15:44–16:00 | Speech layer missing; no visual cue for grammar narration. |
| q008 | the ultimate IELTS playbook | ocr | 05:34 | 16:32–16:50 | OCR layer missing; frame embedding cannot read the cover text. |
| q009 | where he says prove correctness and argue efficiency | speech | 08:42 | 00:15–00:32 | Speech layer missing; the late “measure ops” blackboard looked similar. |
| q010 | where he explains the base case of zero students | speech | 05:24 | 04:48–05:09 | Speech layer missing; 15 s early on the same induction segment. |
| q011 | where he says we measure operations instead of time | speech | 10:38 | 08:25–08:45 | Speech layer missing; repeated “ops” blackboard text. |
| q015 | the chalkboard heading efficiency | ocr | 08:42 | 07:30–08:00 | OCR layer missing; several blackboards have the same heading. |
| q016 | a red umbrella | object | 00:53 | 00:55–01:05 | Small object missed; the wet red pavement dominated the frame. |
| q019 | people crossing the street in the rain | visual | 00:48 | 00:05–00:30 | Scene confusion: a later red umbrella matched “rain” more strongly than the crossing. |
| q020 | a wide road with parked cars | visual | 02:56 | 02:10–02:35 | Repeated road geometry; near miss by 21 s. |
| q023 | a cyclist in a green jacket | object | 05:48 | 01:35–01:45 | Small object missed; another street scene won. |
| q024 | a red car in the distance | object | 06:29 | 00:10–00:20 | Tiny background object; frame embedding could not isolate it. |
| q025 | a zebra crossing on a residential street | visual | 06:53 | 00:05–00:30 | The same crossing repeats later; one valid segment beat the intended one. |
| q027 | cars driving through the intersection | action | 06:59 | 06:30–06:55 | Motion timing: predicted 4 s after the window. |
| q031 | a man sitting at a desk with a laptop | spatial | 03:40 | 02:55–03:15 | Another desk/workspace shot won; no spatial reasoning. |
| q033 | a family playing a game together | caption | 03:43 | 03:45–04:00 | Caption layer missing; 2 s early — visually close but no action grounding. |
| q037 | the registry editor showing current version windows | ocr | 00:40 | 00:50–01:15 | OCR layer missing; the registry window opens 10 s later. |
| q038 | the command sfc /scannow in command prompt | ocr | 02:22 | 02:25–02:50 | OCR layer missing; near miss by 3 s. |
| q042 | a blue Windows desktop | visual | 01:38 | 01:56–02:06 | Repeated blue desktop backgrounds; 18 s early. |

## What this says about Layer A

- **Speech and OCR queries** are close to random without layers D/E, exactly as expected.
- **Object queries** fail when the object is small (umbrella, distant car, cyclist); this is the case for Layer B crops.
- **Visual scene queries** suffer from repeated scenes (roads, crossings, desktops); temporal merging (P4) and captions (P7) should help.
- **Near misses** within a few seconds (q002, q033, q037, q038) show that timestamp precision, not just ranking, is the next problem.
