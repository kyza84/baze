"""SQLAlchemy models."""

from datetime import datetime, timedelta

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, JSON, String
from sqlalchemy.orm import declarative_base

from config import TRIAL_HOURS

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    telegram_id = Column(Integer, unique=True, nullable=False, index=True)
    username = Column(String, nullable=True)
    subscription_until = Column(DateTime, nullable=True)
    is_demo = Column(Boolean, default=False, nullable=False)
    trial_used = Column(Boolean, default=False, nullable=False)
    settings = Column(
        JSON,
        default=lambda: {
            "min_liquidity": 5000,
            "max_age_minutes": 60,
            "alert_cooldown_seconds": 30,
            "notify_enabled": True,
        },
        nullable=False,
    )
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def is_subscribed(self) -> bool:
        return bool(self.subscription_until and self.subscription_until > datetime.utcnow())

    def activate_demo(self) -> None:
        self.is_demo = True
        self.trial_used = True
        self.subscription_until = datetime.utcnow() + timedelta(hours=TRIAL_HOURS)

    def activate_subscription(self, days: float) -> None:
        now = datetime.utcnow()
        start = self.subscription_until if self.subscription_until and self.subscription_until > now else now
        self.subscription_until = start + timedelta(days=float(days))
        self.is_demo = False


class PaymentInvoice(Base):
    __tablename__ = "payment_invoices"

    id = Column(Integer, primary_key=True)
    invoice_id = Column(Integer, unique=True, nullable=False, index=True)
    telegram_id = Column(Integer, nullable=False, index=True)
    plan_type = Column(String, nullable=False)
    amount = Column(Float, nullable=False)
    status = Column(String, default="pending", nullable=False)
    payload = Column(String, nullable=True)
    paid_at = Column(DateTime, nullable=True)
    raw_update = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class ManualPaymentRequest(Base):
    __tablename__ = "manual_payment_requests"

    id = Column(Integer, primary_key=True)
    telegram_id = Column(Integer, nullable=False, index=True)
    username = Column(String, nullable=True)
    plan_type = Column(String, nullable=False)
    amount = Column(Float, nullable=False)
    status = Column(String, default="pending", nullable=False)  # pending/approved/rejected
    comment = Column(String, nullable=True)
    reviewed_by = Column(Integer, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
