# Public benchmark — Charades-STA test subset

## Protocol

- **Dataset:** Charades-STA (Gao et al., 2017), test split annotations from the
  VideoMind mirror of the official labels.
- **Videos:** the official `Charades_v1_480.zip` bundle. Only 50 videos are
  fetched, using HTTP range requests rather than the full 16 GB download.
- **Subset selection:** the 50 test videos with the most queries and at least
  2 queries each. This yields **399 queries** (median 8 per video).
- **Index:** Layer A (SigLIP base, 1 fps, FAISS). No captions, speech, OCR, or
  objects are used for this number.
- **Proposal:** the top-1 frame becomes a **7.5 s** window centred on the peak.
  7.5 s is the rounded median ground-truth duration of the Charades-STA
  **train** split (7.4 s), so it is not tuned on test.
- **Metrics:** standard R@1 IoU 0.5, R@1 IoU 0.7, and mean IoU.

## Results

| Metric | Layer A |
|---|---|
| R@1 IoU 0.5 | **0.421** |
| R@1 IoU 0.7 | **0.241** |
| mIoU | **0.377** |
| Any overlap | 0.692 |

Window sensitivity (same top-1 frames, different proposal lengths):

| Window (s) | R@1 IoU 0.5 | R@1 IoU 0.7 | mIoU |
|---|---|---|---|
| 4.0 | 0.221 | 0.055 | 0.264 |
| 5.0 | 0.308 | 0.130 | 0.309 |
| 6.0 | 0.373 | 0.160 | 0.343 |
| 7.0 | 0.408 | 0.215 | 0.369 |
| 7.5 | 0.421 | 0.241 | 0.377 |
| 8.0 | 0.444 | 0.243 | 0.383 |
| 10.0 | 0.459 | 0.170 | 0.391 |

The default 1-second segment policy scores R@1 IoU 0.5 = 0.008 on these 30 s
clips (`results/charades_sta_a.*`); the fixed-window result is the headline.
This is a frame-retrieval baseline, not a trained temporal-grounding model.
State-of-the-art trained grounders are much higher; compare within the same
subset and protocol.

## Files

| File | Purpose |
|---|---|
| `charades_sta_subset.csv` | 399 queries: `query_id, video_id, query, type, start, end` |
| `charades_sta_subset.json` | Manifest: 50 video ids, query counts, source, CSV SHA-256 |
| `results/charades_sta_a_window75.json/md` | Headline result |
| `results/charades_sta_a.json/md` | Default 1 s segment-policy ablation |

## Reproduce

```bash
python scripts/prepare_charades_sta.py --download --max-videos 50
python scripts/index_public.py
python scripts/evaluate_public.py --window-s 7.5
```

Videos and extracted frames live under `data/public/` and `data/frames/` and
are gitignored; only the CSV, manifest, and result files are versioned.

## Citation

```bibtex
@inproceedings{gao2017tall,
  title={TALL: Temporal Activity Localization via Language Query},
  author={Gao, Jiyang and Sun, Chen and Yang, Zhenheng and Nevatia, Ram},
  booktitle={ICCV},
  year={2017}
}

@inproceedings{sigurdsson2016hollywood,
  title={Hollywood in Homes: Crowdsourcing Data Collection for Activity Understanding},
  author={Sigurdsson, Gunnar A. and Varol, G{\"u}l and Wang, Xiaolong and Laptev, Ivan and Farhadi, Ali and Gupta, Abhinav},
  booktitle={ECCV},
  year={2016}
}
```

The videos are not redistributed by this repository. Follow the Charades and
Charades-STA terms of use.
