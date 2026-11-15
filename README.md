<p align="center">
  <img src="assets/architecture.svg" alt="dormouse — the stack" width="100%">
</p>

<p align="center">
  <a href="https://github.com/dormouse-ai/dormouse/actions/workflows/ci.yml"><img alt="ci" src="https://img.shields.io/github/actions/workflow/status/dormouse-ai/dormouse/ci.yml?branch=master&label=ci"></a>
  <a href="LICENSE"><img alt="license" src="https://img.shields.io/badge/license-Apache--2.0-blue"></a>
  <img alt="python" src="https://img.shields.io/badge/python-3.12%20%E2%80%93%203.15-2a66c2">
</p>

**A language model you own rather than rent.** Weights under Apache-2.0, small
enough to ship as a file, fast enough to run on a laptop, trained on one consumer
GPU so the whole recipe stays checkable.

[Русский](#русский) · [The stack](#the-stack) · [Commands](#commands) · [State](#what-is-real-today) · [Install](#install)

---

## The stack

1.00 B stored, 46 M active, ~250 MB as a file, 0.76 MB/s on one 16 GB card.
Full reasoning per box in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

| slot | choice | why this one |
|---|---|---|
| unit | BPE 32 768, byte fallback | 3.4× fewer steps at equal text than bytes |
| attention | **3 KDA : 1 CSA** per 4 layers | recurrent state with no selector that can miss; sparse layer adds reach |
| FFN | **MoE 64 × 15.34 M + shared, top-2, dropless** | 1 B of capacity at 46 M active |
| activation | **MoA** over six functions, sigmoid gate | −0.029 relative loss vs SwiGLU at 0.12 B–2 B (NeurIPS 2026) |
| optimizer | **Muon** + AdamW | 1.35× fewer steps to the same loss |
| precision | **ternary weights from step 0**, int4 activations | the shipped file is the trained file |
| loss | chunked cross-entropy | removes a 1.6 GB logits tensor (13.3×) |
| head | tied | 27 M parameters not paid twice |

## Commands

```
dorm doctor                    what this machine can do
dorm plan   <config>           what will run, and what it will cost
dorm train  <config> [--yes]   training; past the budget --yes is required
dorm run    <config>           pretrain → sft → eval → export
dorm tokenize <corpus>         text → uint16, with a manifest
dorm eval   <ckpt>             BPB and downstream, one harness
dorm serve  <ckpt>             OpenAI-compatible server
dorm export <ckpt>             quantised package + round-trip check
dorm status [journal]          read a run journal
```

### Why one binary instead of scripts

Three measurements in this project were invalidated by a shell script that exited
`0` after doing nothing, and two more by output piped through `grep`, which hid the
line that mattered. So: no silent success, no surprise cost, and no log grepping —
a run writes a journal and `status` reads it.

```console
$ dorm train
train is not wired yet. Escape: use the current port/ trainer until the model
layer moves; see PLAN.md step 4
$ echo $?
2
```

## What is real today

| | status |
|---|---|
| `doctor`, `status` | working, tested |
| `data/bin.py` | memory-mapped token stream, manifest-checked, resume by arithmetic |
| `metrics.py` | BPB as a type that cannot exist without its evaluation window |
| `model/` | transformer, gated/LA/MoA FFN, attention with RoPE, KDA / CSA / hybrid, MoE bank — each checked against an external reference |
| `train/` | Muon + AdamW, schedules, one step with counters and a NaN firewall |
| `eval`, `sft`, `export`, `serve` | planned; the CLI refuses them by name rather than pretending |

The old research stack lives in [sehaxe/dormouse](https://github.com/sehaxe/dormouse)
as an archive; nothing here imports it.

## Install

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/dorm doctor
.venv/bin/python -m pytest
```

CPU-only torch is enough for the tests:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Design rules

1. **Numbers or it did not happen** — every mechanism ships with the measurement
   that justifies it, or is recorded as a refusal with the number.
2. **External anchors** — a numeric mechanism is checked against a reference
   computed outside this repository. A golden hash from our own run is a lint error.
3. **A negative result is a hypothesis about a bug, not a verdict** — eight checks
   before anything is switched off ([`NEGATIVE_RESULTS.md`](https://github.com/dormouse-ai/plan/blob/master/NEGATIVE_RESULTS.md)).
4. **Hot swap with a receipt** — a component can be replaced at a stage boundary,
   and the swap is a journal event carrying its step number.

## License

Apache-2.0.

---

# Русский

**Языковая модель, которой вы владеете, а не арендуете.** Веса под Apache-2.0,
достаточно маленькие, чтобы быть файлом, достаточно быстрые для ноутбука,
обученные на одной потребительской видеокарте — чтобы весь рецепт можно было
проверить.

## Стек

1.00 млрд хранимых, 46 млн активных, ~250 МБ в файле, 0.76 МБ/с на одной карте
16 ГБ. Полное обоснование каждого блока — в
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

| слот | выбор | почему |
|---|---|---|
| единица | BPE 32 768, байтовый fallback | в 3.4× меньше шагов на том же тексте, чем байты |
| внимание | **3 KDA : 1 CSA** на 4 слоя | рекуррентное состояние без селектора, который может промахнуться; разреженный слой даёт дальнобойность |
| FFN | **MoE 64 × 15.34 млн + общий, top-2, dropless** | 1 млрд ёмкости при 46 млн активных |
| активация | **MoA** по шести функциям, sigmoid-гейт | −0.029 относительного лосса против SwiGLU на 0.12-2 млрд (NeurIPS 2026) |
| оптимизатор | **Muon** + AdamW | в 1.35× меньше шагов до того же лосса |
| точность | **тернар с шага 0**, int4-активации | выпускаемый файл — это и есть обученный файл |
| лосс | chunked CE | убирает тензор логитов на 1.6 ГБ (13.3×) |
| голова | tied | 27 млн параметров не платятся дважды |

## Команды

```
dorm doctor / plan / train / run / tokenize / eval / serve / export / status
```

**Никакого тихого успеха**: команда, которая не подключена, отказывается с
ненулевым кодом и текстом выхода. **Никакой внезапной цены**: то, что обещает
больше бюджета, требует `--yes`. **Никакого поиска по логам**: прогон пишет
журнал, `status` читает его.

## Что реально работает сейчас

`doctor`, `status`, чтение данных с манифестом, BPB как тип, модель (трансформер,
FFN с gated/LA/MoA, внимание с RoPE, KDA / CSA / гибрид, MoE-банк), тренер
(Muon + AdamW, расписания, шаг со счётчиками и NaN-щитом). `eval`, `sft`,
`export`, `serve` — запланированы; CLI отказывает им по имени, а не изображает
работу.

## Правила

1. **Числа или этого не было** — у каждого механизма есть замер, либо отказ с числом.
2. **Внешний якорь** — численная математика сверяется с референсом вне этого
   репозитория; golden-хэш с собственного прогона — ошибка линтера.
3. **Отрицательный результат — гипотеза о баге, а не вердикт** — восемь проверок
   до выключения.
4. **Горячая замена с квитанцией** — замена компонента это событие журнала с
   номером шага.

## Лицензия

Apache-2.0.
