# LLM Assay

**An evaluation harness for measuring LLM quality, latency, and cost on a domain task.**

*Assay* is an analytical-chemistry term — a test that measures the quality and
composition of a substance. This tool does the same for language models: it runs
the same task across GPT, Claude, Gemini, and open Llama-class models, then scores
each one and ranks them by how well they actually perform, how fast they respond,
and what they cost. The built-in task is drawn from pharmaceutical GMP quality
assurance (deviation triage, structured extraction, regulatory Q&A), but the harness
is task-agnostic: point it at your own JSONL dataset and scorers.

It runs **out of the box with no API keys** using deterministic mock providers, so
the whole pipeline — runner, scorers, cost model, and the HTML report — is fully
reproducible in CI and in a demo. Swap in real providers when you want real numbers.

---

## Why this project

Choosing a model for a production feature is a measurement problem, not a vibe.
The questions that matter are: *which model is most accurate on **my** task, how
much latency does it add, and what does each correct answer cost?* This harness
answers all three at once and renders the trade-off — the hero chart is
**quality vs. cost**, because the real engineering decision is "most quality per
dollar," not "highest score regardless of price."

---

## Quickstart

```bash
# 1. Install (Python 3.10+)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Run the default evaluation — no API keys needed (mock providers)
assay run --config config.example.yaml

# 3. Open the report
open reports/report.html        # macOS  (xdg-open on Linux)
```

That produces a terminal leaderboard plus `reports/report.{json,md,html}`.

### Running real models

```bash
cp .env.example .env            # then paste in your keys
export $(grep -v '^#' .env | xargs)   # or use direnv / your shell

# Edit config.example.yaml: comment out the two mock models,
# uncomment the real ones you want, then:
assay run --config config.example.yaml
```

To try the **LLM-as-judge** flow on open-ended Q&A (needs a real key):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
assay run --config config.judge.example.yaml
```

---

## How it works

```
            ┌────────────┐     ┌──────────────────────────────┐
 config ───▶│   Runner   │────▶│  for each (dataset × case ×   │
 (YAML)     │            │     │  model × repeat): call, time, │
            │ concurrency│     │  retry, cache, then score     │
            │ + retries  │     └──────────────┬───────────────┘
            └─────┬──────┘                    │
                  │                  ┌─────────▼─────────┐
        ┌─────────▼────────┐         │     Scorers       │
        │    Providers     │         │ exact / json /    │
        │ anthropic openai │         │ regex / llm_judge │
        │ google hf  mock  │         └─────────┬─────────┘
        └──────────────────┘                   │
                                      ┌─────────▼─────────┐
                                      │     Reporters     │
                                      │ terminal · json · │
                                      │ markdown · html   │
                                      └───────────────────┘
```

Everything talks through two small interfaces — `LLMProvider` and `Scorer` — so
adding a model or a metric means writing one class, never touching the pipeline.

### Design decisions (the "why")

- **Direct REST calls over vendor SDKs.** Each provider is ~80 lines of `httpx`
  against the documented endpoint. *Why:* a tiny, uniform dependency surface, real
  `usage` token counts for honest cost accounting, and the exact API contract stays
  visible instead of hidden behind four different SDK abstractions.
- **A deterministic mock provider is a first-class backend.** Output is a pure
  function of `(seed, prompt)`. *Why:* an eval harness has to be runnable and
  testable without keys or network. The mock lets CI exercise the *entire* pipeline,
  and two seeds produce a genuinely different leaderboard for the demo.
- **The runner owns latency and retries, not the providers.** Each attempt is timed
  with `asyncio.wait_for`; providers just `raise ProviderError(retryable=...)`.
  *Why:* timings are then measured identically across every backend, and retry/backoff
  policy lives in exactly one place.
- **Scorers are async and return a normalised 0–1 score.** *Why:* one uniform
  interface covers cheap string checks and an `llm_judge` that has to make its own
  model call, and normalised scores can be averaged and ranked on the same scale.
  JSON extraction gives **partial credit** (fraction of correct fields) rather than
  all-or-nothing, which is far more informative for extraction tasks.
- **Caching is keyed on `(model, request, repeat)` and never stores errors.** *Why:*
  re-running after editing only the report or scoring config is instant and free,
  while transient failures still get retried next run.
- **Bounded concurrency via a semaphore.** *Why:* saturate throughput without
  tripping provider rate limits.
- **JSON output uses a `{success, message, data}` envelope.** *Why:* consistent with
  the rest of my services, and trivial to consume from another tool or a CI gate.

---

## Reference

### Providers (`backend:model`)

| Backend       | Example spec                                       | Tokens | Notes |
|---------------|----------------------------------------------------|--------|-------|
| `anthropic`   | `anthropic:claude-3-5-sonnet-latest`               | exact  | Messages API |
| `openai`      | `openai:gpt-4o-mini`                               | exact  | Chat Completions |
| `google`      | `google:gemini-1.5-flash`                          | exact  | generateContent |
| `huggingface` | `huggingface:meta-llama/Meta-Llama-3-8B-Instruct` | est.   | Inference API; tokens estimated |
| `mock`        | `mock:smart`                                       | n/a    | Deterministic, free, no network |

### Scorers

| Type               | Use case                       | Key params |
|--------------------|--------------------------------|------------|
| `exact_match`      | Classification / labels        | `ignore_case`, `strip_punct` |
| `normalized_match` | Short answers, format-tolerant | — |
| `contains`         | Must mention a key term        | `value`, `ignore_case` |
| `regex`            | Pattern / format compliance    | `pattern`, `flags` |
| `json_match`       | Structured extraction          | `keys`, `mode` (`value`\|`keys`) — partial credit |
| `llm_judge`        | Open-ended answers             | `threshold`, `criteria` |

### Datasets (JSONL)

One JSON object per line; `#` comments and blank lines are ignored.

| Field        | Required | Meaning |
|--------------|----------|---------|
| `input`      | yes      | The prompt shown to the model |
| `expected`   | —        | Ground truth (string, object, or reference text) |
| `id`         | —        | Stable id (auto-assigned if omitted) |
| `task`       | —        | Hint, e.g. `classification`, `extraction`, `qa` |
| `system`     | —        | Per-case system-prompt override |
| `metadata`   | —        | Free-form |

Three datasets are included: `gmp_severity_classification` (15 cases),
`sop_extraction` (10), and `regulatory_qa` (6, for the judge demo).

### CLI

| Command | Description |
|---------|-------------|
| `assay run --config <file>` | Run an evaluation and write reports |
| `assay providers` | List model backends |
| `assay scorers` | List scorers |
| `assay datasets <file.jsonl>` | Inspect a dataset |
| `assay version` | Print the version |

Useful `run` flags: `--models a,b` (override config), `--limit N` (smoke run),
`--no-cache`, `--format terminal,json,markdown,html`, `--output-dir`, `--stem`.

---

## Reports

- **Terminal** — a ranked leaderboard and per-dataset tables via `rich`.
- **JSON** — full results in a `{success, message, data}` envelope; every record,
  cell, and leaderboard row, ready for a CI quality gate.
- **Markdown** — a leaderboard table to drop straight into a PR or README.
- **HTML** — a self-contained dark "analyzer" report: a quality-vs-cost scatter,
  per-dataset and latency charts, a sortable leaderboard, and a per-case explorer
  with filtering. A sample is in [`examples/sample-report.html`](examples/).

---

## Tests & CI

```bash
ruff check .
pytest -q
```

The suite covers scorers, the cost model, dataset loading, the cache, provider
construction (including the mock's determinism), the reporters, and a full
end-to-end runner pass — all on mock providers, so it needs no keys. CI runs lint
+ tests + a mock smoke-run on Python 3.10–3.12.

---

## Interview talking points

- **Abstraction boundaries that paid off.** Two interfaces (`LLMProvider`, `Scorer`)
  keep the orchestration logic completely model- and metric-agnostic. I can defend
  why the *runner*, not the provider, owns timing and retries: it's the only way to
  get latency numbers that are comparable across vendors with different SDK behaviour.
- **Testability by construction.** Making a deterministic mock a real backend means
  the entire pipeline is exercised in CI with zero secrets and zero flakiness —
  the same trick as splitting `app`/`server` so an HTTP app can be imported in tests
  without binding a port.
- **Honest cost modelling.** I read real `usage` blocks where the API provides them
  and clearly flag where token counts are estimated (HF). Pricing is overridable from
  config because list prices drift — I don't hardcode a number I can't keep current.
- **Concurrency with backpressure.** `asyncio.gather` over a semaphore gives high
  throughput while respecting rate limits; retry uses exponential backoff with jitter
  and only retries errors the provider marks retryable.
- **Partial credit and LLM-judging.** Extraction uses field-level partial credit;
  open-ended answers use an LLM judge, and I can speak to its known biases (verbosity,
  same-family preference) and why judge scores are a strong signal rather than truth.
- **Product framing.** The headline visual is quality-vs-cost, because the decision a
  team actually makes is "most quality per dollar at acceptable latency," not "top of
  a single leaderboard."
- **Domain grounding.** The sample task mirrors real GMP QA work (deviation severity,
  SOP field extraction, regulatory Q&A) from my pharmaceutical-engineering background —
  a concrete example of applying LLM evaluation to a regulated, high-stakes domain.

---

## Caveats

- Mock-provider numbers are **placeholders** that demonstrate the pipeline, not model
  quality. Real comparisons need real API keys.
- Pricing constants are approximate and dated (reviewed early 2025); override them in
  config for accuracy.
- The HTML report loads Chart.js from a CDN; charts need a network connection, but the
  tables render offline.

## License

MIT © Md. Sazed Ul Karim
