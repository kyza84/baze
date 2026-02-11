"""Entry point for Base Alert Bot."""

import asyncio
import logging
import os
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

import config
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, MessageHandler, filters

from bot.handlers import (
    admin_approve_card_command,
    admin_grant_command,
    admin_pending_cards_command,
    admin_reject_card_command,
    admin_stats_command,
    demo_command,
    handle_admin_card_callback,
    handle_text_menu,
    handle_menu_callback,
    handle_payment_callback,
    payment_client,
    settings_command,
    setamount_command,
    setscore_command,
    start_command,
    status_command,
    subscribe_command,
    togglefilter_command,
    test_alert_command,
    mystats_command,
)
from config import APP_LOG_FILE, LOG_LEVEL, LOG_DIR, SCAN_INTERVAL, TELEGRAM_BOT_TOKEN
from database.db import get_all_subscribed_users, get_or_create_user, init_db
from monitor.alerter import TokenAlerter
from monitor.dexscreener import DexScreenerMonitor
from monitor.token_scorer import TokenScorer
from payments.webhook_server import CryptoWebhookServer
from trading.auto_trader import AutoTrader


def configure_logging() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    file_handler = RotatingFileHandler(APP_LOG_FILE, maxBytes=1_000_000, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    # Avoid leaking bot token in verbose transport logs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.INFO)


logger = logging.getLogger(__name__)
AUTO_TRADER_STATE_KEY = "auto_trader_state"
AUTO_TRADER_REASON_KEY = "auto_trader_init_reason"
AUTO_TRADER_LAST_ERROR_KEY = "auto_trader_last_error"
_AUTO_TRADER_INIT_LOCK = asyncio.Lock()


def _merge_source_stats(*parts: dict[str, dict[str, int | float]]) -> dict[str, dict[str, int | float]]:
    merged: dict[str, dict[str, int | float]] = {}
    for block in parts:
        for source, row in (block or {}).items():
            cur = merged.setdefault(
                source,
                {
                    "ok": 0,
                    "fail": 0,
                    "total": 0,
                    "rate_limited": 0,
                    "retries": 0,
                    "error_percent": 0.0,
                    "latency_avg_ms": 0.0,
                    "latency_max_ms": 0.0,
                    "_latency_weighted_sum_ms": 0.0,
                    "_latency_total_samples": 0,
                },
            )
            cur["ok"] = int(cur["ok"]) + int(row.get("ok", 0))
            cur["fail"] = int(cur["fail"]) + int(row.get("fail", 0))
            cur["total"] = int(cur["total"]) + int(row.get("total", 0))
            cur["rate_limited"] = int(cur["rate_limited"]) + int(row.get("rate_limited", 0))
            cur["retries"] = int(cur["retries"]) + int(row.get("retries", 0))
            row_total = int(row.get("total", 0) or 0)
            row_avg = float(row.get("latency_avg_ms", 0.0) or 0.0)
            cur["_latency_weighted_sum_ms"] = float(cur["_latency_weighted_sum_ms"]) + (row_avg * row_total)
            cur["_latency_total_samples"] = int(cur["_latency_total_samples"]) + row_total
            cur["latency_max_ms"] = max(float(cur["latency_max_ms"]), float(row.get("latency_max_ms", 0.0) or 0.0))
    for source, row in merged.items():
        total = int(row.get("total", 0))
        fail = int(row.get("fail", 0))
        row["error_percent"] = round((float(fail) / total * 100.0) if total > 0 else 0.0, 2)
        lat_n = int(row.get("_latency_total_samples", 0) or 0)
        lat_sum = float(row.get("_latency_weighted_sum_ms", 0.0) or 0.0)
        row["latency_avg_ms"] = round((lat_sum / lat_n), 2) if lat_n > 0 else 0.0
        row.pop("_latency_weighted_sum_ms", None)
        row.pop("_latency_total_samples", None)
    return merged


def _policy_state(
    *,
    source_stats: dict[str, dict[str, int | float]],
    safety_stats: dict[str, int | float],
) -> tuple[str, str]:
    checks_total = int(safety_stats.get("checks_total", 0) or 0)
    fail_closed = int(safety_stats.get("fail_closed", 0) or 0)
    api_error_percent = float(safety_stats.get("api_error_percent", 0.0) or 0.0)
    fail_closed_ratio = (float(fail_closed) / checks_total * 100.0) if checks_total > 0 else 0.0
    fail_reason_top = str(safety_stats.get("fail_reason_top", "none") or "none").strip().lower() or "none"
    fail_reason_top_count = int(safety_stats.get("fail_reason_top_count", 0) or 0)

    if bool(getattr(config, "TOKEN_SAFETY_FAIL_CLOSED", False)) and (
        fail_closed_ratio >= float(getattr(config, "DATA_POLICY_FAIL_CLOSED_FAIL_CLOSED_RATIO", 60.0))
        or api_error_percent >= float(getattr(config, "DATA_POLICY_FAIL_CLOSED_API_ERROR_PERCENT", 90.0))
    ):
        return (
            "FAIL_CLOSED",
            (
                f"safety_api_unreliable fail_closed_ratio={fail_closed_ratio:.1f}% "
                f"api_err={api_error_percent:.1f}% top={fail_reason_top}:{fail_reason_top_count}"
            ),
        )

    totals = [float((row or {}).get("error_percent", 0.0) or 0.0) for row in source_stats.values()]
    max_err = max(totals) if totals else 0.0
    if max_err >= float(getattr(config, "DATA_POLICY_DEGRADED_ERROR_PERCENT", 35.0)):
        return "DEGRADED", f"source_errors_high max_err={max_err:.1f}%"

    return "OK", "healthy"


def _format_source_stats_brief(source_stats: dict[str, dict[str, int | float]]) -> str:
    if not source_stats:
        return "none"
    parts: list[str] = []
    for source in sorted(source_stats.keys()):
        row = source_stats.get(source) or {}
        parts.append(
            (
                f"{source}:ok={int(row.get('ok', 0))}"
                f"/fail={int(row.get('fail', 0))}"
                f"/429={int(row.get('rate_limited', 0))}"
                f"/err={float(row.get('error_percent', 0.0)):.1f}%"
                f"/avg={float(row.get('latency_avg_ms', 0.0)):.0f}ms"
            )
        )
    return "; ".join(parts)


def _format_safety_reasons_brief(safety_stats: dict[str, int | float]) -> str:
    counts = safety_stats.get("fail_reason_counts")
    if not isinstance(counts, dict) or not counts:
        return "none"
    pairs: list[tuple[str, int]] = []
    for key, value in counts.items():
        try:
            pairs.append((str(key), int(value)))
        except Exception:
            continue
    if not pairs:
        return "none"
    pairs.sort(key=lambda kv: kv[1], reverse=True)
    return ",".join(f"{k}:{v}" for k, v in pairs[:4])


async def ensure_auto_trader(application: Application, reason: str) -> AutoTrader:
    existing = application.bot_data.get("auto_trader")
    if isinstance(existing, AutoTrader):
        if application.bot_data.get(AUTO_TRADER_STATE_KEY) != "ready":
            application.bot_data[AUTO_TRADER_STATE_KEY] = "ready"
        return existing

    async with _AUTO_TRADER_INIT_LOCK:
        existing = application.bot_data.get("auto_trader")
        if isinstance(existing, AutoTrader):
            if application.bot_data.get(AUTO_TRADER_STATE_KEY) != "ready":
                application.bot_data[AUTO_TRADER_STATE_KEY] = "ready"
            return existing

        application.bot_data[AUTO_TRADER_STATE_KEY] = "initializing"
        application.bot_data[AUTO_TRADER_REASON_KEY] = str(reason)
        application.bot_data["auto_trader_init_at"] = datetime.now(timezone.utc).isoformat()
        logger.info("AUTOTRADER_INIT state=initializing reason=%s", reason)
        try:
            trader = AutoTrader()
        except Exception as exc:
            application.bot_data[AUTO_TRADER_STATE_KEY] = "failed"
            application.bot_data[AUTO_TRADER_LAST_ERROR_KEY] = str(exc)
            logger.exception("AUTOTRADER_INIT state=failed reason=%s err=%s", reason, exc)
            raise

        application.bot_data["auto_trader"] = trader
        application.bot_data[AUTO_TRADER_STATE_KEY] = "ready"
        application.bot_data[AUTO_TRADER_LAST_ERROR_KEY] = ""
        trader.set_data_policy("OK", "init_reset")
        logger.info(
            "AUTOTRADER_INIT state=ready reason=%s mode=%s",
            reason,
            ("paper" if bool(config.AUTO_TRADE_PAPER) else "live"),
        )
        return trader


async def monitoring_loop(application: Application) -> None:
    monitor = DexScreenerMonitor()
    alerter = TokenAlerter()
    scorer = TokenScorer()
    auto_trader = await ensure_auto_trader(application, reason="restart")

    try:
        while True:
            try:
                tokens = await monitor.fetch_new_tokens()
                if tokens:
                    if config.PERSONAL_MODE and int(config.PERSONAL_TELEGRAM_ID or 0) > 0:
                        users = [get_or_create_user(int(config.PERSONAL_TELEGRAM_ID), None)]
                    else:
                        users = get_all_subscribed_users()

                    if users:
                        high_quality = 0
                        alerts_sent = 0
                        trade_candidates: list[tuple[dict, dict]] = []
                        for token in tokens:
                            score_data = scorer.calculate_score(token)
                            token["score_data"] = score_data

                            if int(score_data.get("score", 0)) >= 70:
                                high_quality += 1

                            if config.AUTO_FILTER_ENABLED and int(score_data.get("score", 0)) < int(config.MIN_TOKEN_SCORE):
                                continue

                            # Trading candidates should not depend on Telegram delivery success.
                            trade_candidates.append((token, score_data))
                            sent_count = await alerter.send_alert(application.bot, token, users, score_data=score_data)
                            alerts_sent += sent_count

                        source_stats = _merge_source_stats(monitor.runtime_stats(reset=True))
                        safety_stats = alerter.runtime_stats(reset=True)
                        policy_state, policy_reason = _policy_state(
                            source_stats=source_stats,
                            safety_stats=safety_stats,
                        )
                        auto_trader.set_data_policy(policy_state, policy_reason)
                        if policy_state == "FAIL_CLOSED":
                            logger.warning(
                                "SAFETY_MODE fail_closed active; BUY disabled until safety API recovers. reason=%s",
                                policy_reason,
                            )
                        elif policy_state == "DEGRADED":
                            logger.warning(
                                "DATA_MODE degraded; BUY disabled by policy. reason=%s",
                                policy_reason,
                            )

                        opened_trades = 0
                        if trade_candidates and policy_state == "OK":
                            opened_trades = await auto_trader.plan_batch(trade_candidates)
                        elif trade_candidates:
                            logger.warning(
                                "AUTO_POLICY mode=%s action=no_buy reason=%s candidates=%s",
                                policy_state,
                                policy_reason,
                                len(trade_candidates),
                            )

                        logger.info(
                            "Scanned %s tokens | High quality: %s | Alerts sent: %s | Trade candidates: %s | Opened: %s | Policy: %s(%s) | Safety reasons: %s | Sources: %s",
                            len(tokens),
                            high_quality,
                            alerts_sent,
                            len(trade_candidates),
                            opened_trades,
                            policy_state,
                            policy_reason,
                            _format_safety_reasons_brief(safety_stats),
                            _format_source_stats_brief(source_stats),
                        )
                await auto_trader.process_open_positions(application.bot)
            except Exception:
                logger.exception("Monitoring loop error")

            await asyncio.sleep(SCAN_INTERVAL)
    finally:
        auto_trader.flush_state()
        await auto_trader.shutdown("telegram_monitor_shutdown")
        await monitor.close()
        await alerter.close()


async def post_init(application: Application) -> None:
    await ensure_auto_trader(application, reason="cold_start")
    application.bot_data["monitor_task"] = asyncio.create_task(monitoring_loop(application))

    webhook_server = CryptoWebhookServer(payment_client)
    await webhook_server.start()
    application.bot_data["webhook_server"] = webhook_server


async def post_shutdown(application: Application) -> None:
    task = application.bot_data.get("monitor_task")
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    webhook_server: CryptoWebhookServer | None = application.bot_data.get("webhook_server")
    if webhook_server:
        await webhook_server.stop()

    await payment_client.close()


def main() -> None:
    configure_logging()
    init_db()

    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).post_init(post_init).post_shutdown(post_shutdown).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("demo", demo_command))
    app.add_handler(CommandHandler("subscribe", subscribe_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("settings", settings_command))
    app.add_handler(CommandHandler("testalert", test_alert_command))
    app.add_handler(CommandHandler("mystats", mystats_command))
    app.add_handler(CommandHandler("setscore", setscore_command))
    app.add_handler(CommandHandler("setamount", setamount_command))
    app.add_handler(CommandHandler("togglefilter", togglefilter_command))
    app.add_handler(CommandHandler("admin_stats", admin_stats_command))
    app.add_handler(CommandHandler("admin_grant", admin_grant_command))
    app.add_handler(CommandHandler("admin_pending_cards", admin_pending_cards_command))
    app.add_handler(CommandHandler("admin_approve_card", admin_approve_card_command))
    app.add_handler(CommandHandler("admin_reject_card", admin_reject_card_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_menu))

    app.add_handler(CallbackQueryHandler(handle_admin_card_callback, pattern=r"^(approve_card_|reject_card_)"))
    app.add_handler(CallbackQueryHandler(handle_payment_callback, pattern=r"^(pay_|card_)"))
    app.add_handler(
        CallbackQueryHandler(
            handle_menu_callback,
            pattern=r"^(demo|subscribe|status|settings|main_menu|toggle_notify|toggle_auto_trade|toggle_auto_filter|set_liquidity.*|set_max_age.*|set_interval.*|skip_.*)$",
        )
    )

    app.run_polling()


if __name__ == "__main__":
    main()

