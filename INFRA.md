# Infrastructure — dormouse

Цель: как у DeepSeek — маленькие репозитории с одной ответственностью, замороженным API и своим CI. Не один репозиторий на всё.

## Репозитории

| репозиторий | что | зависит от |
|---|---|---|
| `dormouse-core` | `Spec`, `Coeffects`, `Counters`, ошибки, реестр, протокол плагина | — |
| `dormouse-kernels` | Triton-ядра: KDA, CSA/HCA, Muon, A4, TSCT | — |
| `dormouse` | данные, модель, обучение, eval, serve, CLI | core, kernels |
| `dormouse-bench` | бенчмарки, профили, A/B | core, kernels, dormouse |
| `dormouse-data` | манифесты, токенизатор, скрипты сборки; **не** бинарники | — |
| `dormouse-models` | конфиги, карточки весов; веса — на Hugging Face | dormouse |
| `dormouse-serve` | HTTP/OpenAI-compatible API, если выделять | dormouse |

**Правило:** репозиторий отделяется, когда у него свой потребитель, свой темп релизов или потребность заморозиться. Всё, что меняется в одном коммите, живёт вместе.

## Границы

```
core      ← kernels? нет
kernels   ← core (counters/errors)
dormouse  → core, kernels
bench     → dormouse
serve     → dormouse
models    → dormouse
data      → нет кода
```

`dormouse` не содержит Triton. `core` не импортирует `dormouse`. `kernels` не импортирует `dormouse`.

Проверка: import-linter + AST-тест в CI.

## Версии

- `core` и `kernels`: semver, тег `vX.Y.Z`, `api-frozen` job.
- `dormouse`: пинится на теги `core`/`kernels`.
- Ломающее изменение API — новая major версия, не «потом поправим».

## CI

Каждый репозиторий:

1. `ruff check .`
2. `ruff format --check .`
3. `mypy src`
4. `pytest --cov --cov-report=term-missing`
5. `api-frozen` для core/kernels
6. `import-linter`
7. wheel + smoke install

GPU-тесты — отдельный self-hosted runner, не на PR.

## Релизы

- тег → GitHub Release → wheel/sdist;
- `dormouse` — Docker image;
- веса — Hugging Face, не в git;
- данные — манифесты + SHA, не бинарники.

## Что запрещено

- большие `.bin` в git (GH001);
- `git add .`;
- push без зелёного CI;
- один репозиторий на всё;
- дублирование ядер в `dormouse`;
- default-off флаги.

## Текущее состояние

- `app` — `dormouse`, 3.5 ГБ `.git`, в истории два больших `.bin`;
- `core` — есть;
- `kernels` — есть;
- чистый push требует **новой истории** или **orphan-ветки**: старый `app` не пушится (GH001).

## Первый шаг

1. Закоммитить P0 в `arch/csa-hca`.
2. Собрать **clean orphan** `arch/csa-hca-clean` без больших файлов.
3. Push clean-ветки в `dormouse-ai/dormouse`; PR.
4. Не force-push `master`.
