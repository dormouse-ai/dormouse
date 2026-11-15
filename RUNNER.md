# The GPU runner

`gpu.yml` is the only workflow that needs the card. It does not run on pull requests:
a contributor's PR must never block on, or hold, a GPU. It runs on a push to `master`
and by hand.

A job with `runs-on: [self-hosted, gpu]` and no registered runner **queues forever**,
which is worse than failing. So the job is gated twice:

| gate | where | when the gate is closed |
|---|---|---|
| `vars.GPU_RUNNER_ONLINE == 'true'` | job `if:` | the job is *skipped* in seconds — a job whose `if` is false is never assigned to a runner |
| a consent marker file | first step | the steps are skipped, the run page says `gpu gate skipped`, and a warning is annotated |

The variable says "a runner exists". The marker says "this host offers its card right
now". Two different questions, and both must be yes.

## One-time registration

On the host with the card — 16 GB, `sm_120` here — as the user that will own the
runner:

```bash
# 1. A registration token, from a machine with `gh` and admin on the repo.
#    It expires in one hour; re-run this line if the rest takes longer.
gh api -X POST repos/dormouse-ai/dormouse/actions/runners/registration-token --jq .token

# 2. On the runner host: the runner of this exact version, hash from its release notes.
RUNNER_VERSION=2.338.2
mkdir -p ~/actions-runner-dormouse && cd ~/actions-runner-dormouse
curl -fsSLO "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"
echo "baebf204307bac6193480852732f0aba2ae97b6c5597c244f4d5558390147c40  actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz" | sha256sum -c -
tar xzf "actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"

# 3. Register. The label is `gpu`; `self-hosted` is always present.
./config.sh \
  --url https://github.com/dormouse-ai/dormouse \
  --token "<the token from step 1>" \
  --name "rtx-5070-ti" \
  --labels gpu \
  --unattended --replace

# 4. Run it. `svc.sh` survives a reboot; `./run.sh` is the foreground version, which
#    is what to use the first time, so a failed install prints to a terminal.
sudo ./svc.sh install "$USER" && sudo ./svc.sh start
#   or: ./run.sh
```

Then the two gates:

```bash
# The host offers the card to CI. Without this file the job skips with a warning.
touch ~/.dormouse-gpu-ci

# A runner exists for this label. Without this the job is skipped, not queued.
gh variable set GPU_RUNNER_ONLINE --body true -R dormouse-ai/dormouse
```

`dormouse-core` has no torch and no card to test, so it has no GPU job and does not
need the runner.

## Serialization

Two GPU jobs must never overlap, and there are two layers because there are two
failure modes:

* **Within a repo**, `gpu.yml`'s `concurrency: group: gpu` queues a second run behind
  the first instead of cancelling it.
* **Across repos and across everything else**, the runner *is* the lock: one
  self-hosted runner takes exactly one job at a time. Do not register two `gpu`
  runners on one card to "make it faster" — that is the same card.

A human training on the host is not covered by either, which is what the marker file
is for: remove `~/.dormouse-gpu-ci` before a long run of your own, and put it back
after. A job that arrives while the marker is gone skips with a visible warning rather
than competing for VRAM.

## What the job proves

`tools/gpu_check.py`, four checks, each a defect this project has actually shipped:

1. **20 steps** of the live recipe in fp32: every loss finite, and every mechanism the
   stack declares enabled reports a non-zero counter in the step that ran it. A
   silently absent mechanism raises `MissingReport` from inside `train_step`.
2. **bf16 against fp32** on the first step, same seed, same weights, within 1%. A
   canary matmul inside the same autocast must return `bfloat16`: without it, a silent
   fp32 fallback would *pass* the tolerance, because falling back makes the match
   closer, not worse.
3. **a Triton kernel** compiled and called through the toolchain `@triton.jit` kernels
   in `model/` use, so an unsupported arch fails in `ptxas` and not at 3 a.m.
4. **peak VRAM** against a 12 GiB ceiling on a 16 GB card, printed to the job summary
   whether the job passes or fails.

Exit 1 with every failed assertion printed. A ceiling is a stated number, not a
feeling: if a batch that "should" fit stops fitting, this is where it says so first.
