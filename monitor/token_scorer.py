"""Token scoring logic for personal trading mode."""

from typing import Any


class TokenScorer:
    def calculate_score(self, token_data: dict[str, Any]) -> dict[str, Any]:
        liquidity = float(token_data.get("liquidity") or 0)
        volume_5m = float(token_data.get("volume_5m") or 0)
        risk_level = str(token_data.get("risk_level") or "MEDIUM").upper()
        age_seconds = int(token_data.get("age_seconds") or 0)
        warning_flags = int(token_data.get("warning_flags") or 0)
        is_contract_safe = bool(token_data.get("is_contract_safe", True))
        source = str(token_data.get("source") or "").strip().lower()
        abs_change_5m = abs(float(token_data.get("price_change_5m") or 0.0))

        liquidity_score = self._score_liquidity(liquidity)
        volume_score = self._score_volume(volume_5m)
        risk_score = self._score_risk(risk_level)
        age_score = self._score_age(age_seconds)

        safety_score = self._score_safety(warning_flags, is_contract_safe)
        score = max(0, min(100, liquidity_score + volume_score + risk_score + age_score + safety_score))

        # Non-watch uplift:
        # strong, safer non-watch candidates should not be permanently stuck at HOLD=60.
        non_watch_uplift = self._score_non_watch_uplift(
            source=source,
            liquidity=liquidity,
            volume_5m=volume_5m,
            age_seconds=age_seconds,
            abs_change_5m=abs_change_5m,
            risk_level=risk_level,
            warning_flags=warning_flags,
            is_contract_safe=is_contract_safe,
        )
        score = max(0, min(100, int(score) + int(non_watch_uplift)))
        recommendation = self._recommendation(score)

        return {
            "score": score,
            "breakdown": {
                "liquidity_score": liquidity_score,
                "volume_score": volume_score,
                "risk_score": risk_score,
                "age_score": age_score,
                "safety_score": safety_score,
                "non_watch_uplift": int(non_watch_uplift),
            },
            "recommendation": recommendation,
        }

    @staticmethod
    def _score_liquidity(liquidity: float) -> int:
        if liquidity >= 20000:
            return 30
        if liquidity >= 10000:
            return 20
        if liquidity >= 5000:
            return 10
        return 0

    @staticmethod
    def _score_volume(volume_5m: float) -> int:
        if volume_5m >= 10000:
            return 30
        if volume_5m >= 5000:
            return 20
        if volume_5m >= 2000:
            return 10
        return 0

    @staticmethod
    def _score_risk(risk_level: str) -> int:
        if risk_level == "LOW":
            return 25
        if risk_level == "MEDIUM":
            return 15
        return 0

    @staticmethod
    def _score_age(age_seconds: int) -> int:
        # New pairs are where most scams live. Reward some maturation instead of "newness".
        if age_seconds < 180:
            return -10
        if age_seconds < 600:
            return 0
        if age_seconds < 1800:
            return 5
        if age_seconds < 3600:
            return 10
        return 15

    @staticmethod
    def _score_safety(warning_flags: int, is_contract_safe: bool) -> int:
        score = 0
        if is_contract_safe:
            score += 8
        if warning_flags <= 0:
            score += 7
        elif warning_flags == 1:
            score += 2
        else:
            score -= min(15, warning_flags * 4)
        return score

    @staticmethod
    def _score_non_watch_uplift(
        *,
        source: str,
        liquidity: float,
        volume_5m: float,
        age_seconds: int,
        abs_change_5m: float,
        risk_level: str,
        warning_flags: int,
        is_contract_safe: bool,
    ) -> int:
        src = str(source or "").strip().lower()
        if (not src) or src.startswith("watchlist"):
            return 0
        if not is_contract_safe:
            return 0
        if int(warning_flags) > 1:
            return 0
        if str(risk_level).upper() not in {"LOW", "MEDIUM"}:
            return 0

        # Conservative uplift only for mature, liquid and not-overheated non-watch tokens.
        # This path helps non-watch avoid permanent HOLD=60 when safety is clean but 5m volume is modest.
        if (
            liquidity >= 100000
            and volume_5m >= 600
            and age_seconds >= 300
            and abs_change_5m <= 5.0
            and int(warning_flags) == 0
        ):
            return 30
        if (
            liquidity >= 70000
            and volume_5m >= 250
            and age_seconds >= 240
            and abs_change_5m <= 6.0
            and int(warning_flags) == 0
        ):
            return 25
        if (
            liquidity >= 45000
            and volume_5m >= 1200
            and age_seconds >= 300
            and abs_change_5m <= 12.0
        ):
            return 25
        if (
            liquidity >= 30000
            and volume_5m >= 2500
            and age_seconds >= 240
            and abs_change_5m <= 10.0
        ):
            return 20
        if (
            liquidity >= 20000
            and volume_5m >= 5000
            and age_seconds >= 240
            and abs_change_5m <= 8.0
        ):
            return 15
        return 0

    @staticmethod
    def _recommendation(score: int) -> str:
        if score >= 70:
            return "BUY"
        if score >= 50:
            return "HOLD"
        return "SKIP"
