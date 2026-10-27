# cleanup

LangGraph pipeline that filters low-quality images out of a YOLO dataset before training
(RF-DETR / YOLO). Source data is never modified. Passing items are copied to a clean output dir.

```
load_dataset -> assess_images -> [judge_borderline] -> write_output -> write_report
```

- **assess_images**: deterministic checks (unreadable, resolution, exposure, blur via Laplacian
  variance, YOLO label validity).
- **judge_borderline**: only runs when some image is borderline (blur just above threshold).
  Pluggable `Judge`. Default applies `borderline_policy`. Optional free local VLM judge via
  Ollama (`--judge-model`). With `--judge-scope survivors` the VLM also reviews every image that
  passed the cheap checks.

## Dataset layout

```
data/raw/
  images/{train,val}/*.jpg
  labels/{train,val}/*.txt   # class cx cy w h (normalized)
```

## Usage

```bash
make install
make dry-run DATA=data/raw          # report only -> data/clean/report.json
make run DATA=data/raw OUT=data/clean
make stats
make check                          # lint + typecheck + tests
```

## VLM judge (free, local)

```bash
brew install ollama
make ollama-serve                   # separate terminal
make ollama-pull VLM=qwen2.5vl:7b
make run-vlm DATA=data/raw SCOPE=survivors
```

VLM reasons land in `report.json` as `judge:keep|drop:<reason>`. If the model errors, the image
falls back to `borderline_policy` with reason `judge_error:<Exception>`.

Progress bars (tqdm) show per-stage progress with a live pass/fail tally; structlog prints a
summary per stage. `CLEANUP_LOG_LEVEL=DEBUG make run-vlm` also logs each judge decision.

Thresholds: see `src/cleanup/config.py`; override with `CLEANUP_*` env vars (`.env.example`).
