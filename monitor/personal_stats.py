"""In-memory personal alert statistics."""

from datetime import datetime

_stats: dict[int, dict[str, int | str]] = {}


def _today_str() -> str:
    return datetime.utcnow().date().isoformat()


def _ensure(user_id: int) -> dict[str, int | str]:
    if user_id not in _stats:
        _stats[user_id] = {
            "last_date": _today_str(),
            "alerts_today": 0,
            "high_score_today": 0,
            "total_alerts": 0,
            "total_high_score": 0,
        }
    return _stats[user_id]


def record_alert(user_id: int, score: int) -> None:
    if user_id <= 0:
        return
    data = _ensure(user_id)
    today = _today_str()
    if data["last_date"] != today:
        data["last_date"] = today
        data["alerts_today"] = 0
        data["high_score_today"] = 0

    data["alerts_today"] = int(data["alerts_today"]) + 1
    data["total_alerts"] = int(data["total_alerts"]) + 1

    if score >= 70:
        data["high_score_today"] = int(data["high_score_today"]) + 1
        data["total_high_score"] = int(data["total_high_score"]) + 1


def get_stats(user_id: int) -> dict[str, int]:
    data = _ensure(user_id)
    today = _today_str()
    if data["last_date"] != today:
        data["last_date"] = today
        data["alerts_today"] = 0
        data["high_score_today"] = 0

    return {
        "alerts_today": int(data["alerts_today"]),
        "high_score_today": int(data["high_score_today"]),
        "total_alerts": int(data["total_alerts"]),
        "total_high_score": int(data["total_high_score"]),
    }
