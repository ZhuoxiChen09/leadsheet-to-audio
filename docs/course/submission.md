# Course presentation and submission

## Verify the HTML export

Verify:

```powershell
python -c "print(open('workbook.html', encoding='utf-8').read(20))"
```

```bash
python -c "print(open('workbook.html', encoding='utf-8').read(20))"
```

Should print `<!DOCTYPE html>\n\n<ht`.

## 7. Record the 20-minute presentation

Suggested structure (from the assignment spec):

| Time | Section |
|---|---|
| 0:00–1:00 | Intro & end-to-end framing |
| 1:00–2:30 | T2 — Data + EDA |
| 2:30–5:00 | T2 — Modeling |
| 5:00–7:30 | T2 — Evaluation |
| 7:30–8:30 | T2 — Related work |
| 8:30–10:00 | T4 — Data + EDA |
| 10:00–13:00 | T4 — Modeling |
| 13:00–16:00 | T4 — Evaluation |
| 16:00–17:00 | T4 — Related work |
| 17:00–18:30 | End-to-end demo |
| 18:30–20:30 | Play the music (doesn't count) |

Recording: OBS Studio or Zoom-record-and-export. Upload as **mp4** to Google
Drive with "anyone with the link" sharing, copy the link into
`video_url.txt`. Or use a YouTube unlisted link.

```powershell
# Smoke-check the video URL passes the autograder's checks
python -c "import requests; url=open('video_url.txt').read().strip(); print(url, requests.head(url, allow_redirects=True))"
```

```bash
# Smoke-check the video URL passes the autograder's checks
python -c "import requests; url=open('video_url.txt').read().strip(); print(url, requests.head(url, allow_redirects=True))"
```

## 8. Gradescope submission

Upload the four files (plus the optional ones if desired):

* `workbook.html` (must start with `<!DOCTYPE html>`)
* `video_url.txt` (one line)
* `symbolic_conditioned.mid` (Task 2 output)
* `continuous_conditioned.mp3` (Task 4 output, > 0 bytes)

The autograder verifies (a) the HTML preamble; (b) the video is downloadable,
mp4, >1 MB; (c) you submitted files for two tasks. There is no quality check —
peer graders evaluate the notebook + presentation against the rubric.
