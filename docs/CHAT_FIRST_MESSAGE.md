# CHAT_FIRST_MESSAGE

Скопируй блок ниже целиком в первый запрос нового чата (при переходе между аккаунтами/сессиями).

```text
Ты работаешь в репозитории d:\\earnforme\\solana-alert-bot (remote: https://github.com/kyza84/baze, branch: main).

Режим работы:
- Только evidence-based инженерные действия.
- Ничего не ломать и не трогать торговую стратегию без фактов.
- Перед любыми правками: диагностика 30m/60m (funnel + причины + runtime tuner actions).
- Любое изменение оформлять как: проблема -> решение -> риск -> проверка.

Прочитай в таком порядке:
1) docs/PROJECT_STATE.md
2) docs/OPERATOR_PATTERNS.md
3) docs/ACCOUNT_TRANSFER_PROMPT.md
4) data/matrix/runs/active_matrix.json
5) последние логи профиля logs/matrix/u_station_ab_night_autotune_v2/

Текущий профиль:
- u_station_ab_night_autotune_v2

Первый ответ дай строго в формате:
1) Текущее состояние процессов (matrix/tuner/watchdog/sync)
2) Воронка 30m и 60m (raw->pre->plan->open->close)
3) Топ-5 reason_code по этапам
4) Узкое место (1 primary bottleneck)
5) План 1-2 безопасных шагов без изменения anti-scam hard-guard

Запрещено:
- destructive git
- отключать hard safety / anti-scam
- делать большие переписывания без локальных доказательств

Отдельно:
- Если пользователь пишет "только ответ" или "ничего не меняй" — соблюдать это строго.
- Команды из docs/OPERATOR_PATTERNS.md трактовать как контракт исполнения.
```

## Правило актуальности
- Файл `docs/CHAT_FIRST_MESSAGE.md` должен обновляться в каждом material-коммите вместе с `docs/PROJECT_STATE.md`.
- Проверяется pre-commit guard (`tools/context_commit_guard.py`, `.githooks/pre-commit`).
