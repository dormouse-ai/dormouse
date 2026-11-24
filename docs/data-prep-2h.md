# data-prep-2h: a valid token stream for the 2h run

Worktree: `wt/data-prepare` @ 27bd9e0. Nothing here is committed.

## Why the first attempt was garbage

`bytes_per_token=1.0` on a uint16 BPE stream is a false denominator, not a data
measurement: `loss 722.76 / 1.0 / ln2 = 1042.7`, the reported "bpb 1030". The same
loss divided by the measured `4.188` is `249` — still a random-init loss, not a data
verdict. `research/dbg/02-data-bugs.md` also found the deeper P0: the root
`tokenizer32k.json` hashes to `ed73858678316df3` and is **legacy** (3 added tokens,
ordinary ids start at 3). `data.tokenizer.load()` here does not call
`check_specials`, so that file passes `check_vocab(32768)` while the model reads
id 4 as `<|end|>` — a silent id-space conflict.

This prep uses the **NEW** 14-special tokenizer `50235922e0bacb3a` (copied from
`wt/engine-mainline/tokenizer32k.json` into `data-prep-2h/tokenizer32k.json`) and
tokenizes straight into its id space. No legacy→NEW `+14` conversion, no uint8
overflow (`242..255 → 0..13`), no empty `tokenizer_sha`.

## Build (exact commands)

```sh
cd /home/sehaxe/dormouse2/wt/data-prepare
cp /home/sehaxe/dormouse2/wt/engine-mainline/tokenizer32k.json data-prep-2h/tokenizer32k.json
PYTHONPATH=src /home/sehaxe/dormouse2/app/.venv/bin/python scripts/build_data_prep_2h.py \
  --shard /mnt/e43497ab-0ff2-45b4-b45f-28de3339a53e/dormouse-datasets/dclm/global-shard_01_of_10/local-shard_0_of_10/shard_00000090_processed.jsonl.zst \
  --tokenizer data-prep-2h/tokenizer32k.json \
  --out-dir data-prep-2h --target-bytes 32000000
```

The script refuses a tokenizer that fails `check_specials`, writes ids as `uint16`,
writes `sha_file(tokenizer)` into every manifest, measures
`bytes_per_token = sum(len(doc.encode()) for doc in docs) / len(ids)` on exactly the
documents it tokenized, writes `*.docs.json` hashes, and refuses to write when the
train/held-out document-hash intersection is non-empty (measured, not a literal 0).
The shard was split by document: 4579 train, 1145 held-out, intersection 0.

## Results

| file | tokens | tokenizer_sha | bytes_per_token | bpb |
|---|---:|---|---:|---:|
| `data-prep-2h/train.bin` | 6,150,993 | `50235922e0bacb3a` | 4.187927 | 2.799 (train loss, step 199) |
| `data-prep-2h/heldout.bin` | 1,503,331 | `50235922e0bacb3a` | 4.165858 | **2.7142** (held-out, 200-step check) |

Both files are private document sets from the same DCLM shard. `bin.Stream` reads
both without refusal; manifest bytes equal `tokens * 2`. Ids: min 14, max 32767,
**zero** ids below 14 — proof the stream is in the NEW id space (legacy would show
ids 3..13 as ordinary). No id ≥ 32768.

## Validation (exact commands)

```sh
# reader gate
PYTHONPATH=src /home/sehaxe/dormouse2/app/.venv/bin/python -c '
from dormouse.data.bin import Stream
s=Stream("data-prep-2h/train.bin"); print(s.tokens, s.manifest.tokenizer_sha, s.take(0,16))'

# plan
PYTHONPATH=src /home/sehaxe/dormouse2/app/.venv/bin/dorm plan data-prep-2h/plan.toml

# 2-step smoke — reports bpb 11.13 by design: random-init CE of a 47k stack, not a data number
PYTHONPATH=src /home/sehaxe/dormouse2/app/.venv/bin/dorm train \
  data-prep-2h/smoke.toml --smoke --steps 2 --runs /tmp/opencode/dorm2h_smoke_new

# the BPB gate: 200 steps on the regression stack, then the held-out window
PYTHONPATH=src /home/sehaxe/dormouse2/app/.venv/bin/python scripts/stream_check.py \
  --train data-prep-2h/train.bin --heldout data-prep-2h/heldout.bin --steps 200
```

Convergence trace: `loss 62.61 → 15.81 → 10.09 → 8.70 → 8.12`;
`heldout_loss 7.8789`, `bytes_per_token 4.187927`, `heldout_bpb 2.7142`
(`bpb_in_band=true`, `learned=true`). That number sat on the repo's regression
baseline at the time (now superseded, see [`regression-baselines.md`](regression-baselines.md))
without any fixture or assumed byte ratio.

The 2-step `--smoke` command is kept as the seam check: it exercises manifest,
tokenizer gate, mixture, journal and eval-window wiring. Its BPB is not a quality
number — at init the tied head's N(0,1) rows over 32768 ids give ~32 nats/token on
the tiny stack (and ~700 on the 401M stack), so any claim of "bpb 2-4 in 2 steps"
would be measuring the init, not the data. Quality is measured by the 200-step check.

## Follow-ups (not done here)

- `data/tokenizer.py::tokenize` still concatenates documents without `END_ID`
  (DBG2 #13): no stop token is learned from this pretrain stream. Fix in the corpus
  path, not in SFT, and re-measure BPT if adopted.
- `runconfig.check_tokenizer` in this worktree checks sources but not `eval`
  (DBG2 #11), and does not require `dtype=uint16` for `unit=bpe` (DBG2 #12). Our
  manifests are valid; the gate itself is still loose.
- `dorm plan` prices this 401M config at ~20min on the *active* FLOP bill, while the
  config comment quotes 1.965h on the conservative stored-parameter bill. Out of
  scope for the data fix.
