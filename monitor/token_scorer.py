"""Token scoring logic for personal trading mode."""

from typing import Any


class TokenScorer:
    def calculate_score(self, token_data: dict[str, Any]) -> dict[str, Any]:
        liquidity = float(token_data.get("liquidity") or 0)
        volume_5m = float(token_data.get("volume_5m") or 0)
        risk_level = str(token_data.get("risk_level") or "MEDIUM").upper()
        age_seconds = int(token_data.get("age_seconds") or 0)

        liquidity_score = self._score_liquidity(liquidity)
        volume_score = self._score_volume(volume_5m)
        risk_score = self._score_risk(risk_level)
        age_score = self._score_age(age_seconds)

        score = max(0, min(100, liquidity_score + volume_score + risk_score + age_score))
        recommendation = self._recommendation(score)

        return {
            "score": score,
            "breakdown": {
                "liquidity_score": liquidity_score,
                "volume_score": volume_score,
                "risk_score": risk_score,
                "age_score": age_score,
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
        if age_seconds < 300:
            return 15
        if age_seconds < 600:
            return 10
        if age_seconds < 1800:
            return 5
        return 0

    @staticmethod
    def _recommendation(score: int) -> str:
        if score >= 70:
            return "BUY"
        if score >= 50:
            return "HOLD"
        return "SKIP"
