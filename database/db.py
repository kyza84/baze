"""Database helpers and CRUD operations."""

from datetime import datetime
from typing import Optional

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from config import DATABASE_URL
from database.models import Base, ManualPaymentRequest, PaymentInvoice, User

engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    _apply_runtime_migrations()


def _apply_runtime_migrations() -> None:
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())
    if "users" not in table_names:
        return

    user_columns = {col["name"] for col in inspector.get_columns("users")}
    if "trial_used" not in user_columns:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE users ADD COLUMN trial_used BOOLEAN NOT NULL DEFAULT 0"))
            # Mark existing demo users as already-used trial.
            conn.execute(text("UPDATE users SET trial_used = 1 WHERE is_demo = 1"))


def get_db() -> Session:
    return SessionLocal()


def get_user(telegram_id: int) -> Optional[User]:
    db = get_db()
    try:
        return db.query(User).filter(User.telegram_id == telegram_id).first()
    finally:
        db.close()


def create_user(telegram_id: int, username: Optional[str]) -> User:
    db = get_db()
    try:
        user = User(telegram_id=telegram_id, username=username)
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    finally:
        db.close()


def get_or_create_user(telegram_id: int, username: Optional[str]) -> User:
    db = get_db()
    try:
        user = db.query(User).filter(User.telegram_id == telegram_id).first()
        if user:
            if username and user.username != username:
                user.username = username
                db.commit()
                db.refresh(user)
            return user

        user = User(telegram_id=telegram_id, username=username)
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    finally:
        db.close()


def get_all_subscribed_users() -> list[User]:
    db = get_db()
    try:
        now = datetime.utcnow()
        return db.query(User).filter(User.subscription_until.is_not(None), User.subscription_until > now).all()
    finally:
        db.close()


def update_user_settings(telegram_id: int, settings: dict) -> Optional[User]:
    db = get_db()
    try:
        user = db.query(User).filter(User.telegram_id == telegram_id).first()
        if not user:
            return None

        current = dict(user.settings or {})
        current.update(settings)
        user.settings = current
        db.commit()
        db.refresh(user)
        return user
    finally:
        db.close()


def update_user_subscription(telegram_id: int, days: float) -> Optional[User]:
    db = get_db()
    try:
        user = db.query(User).filter(User.telegram_id == telegram_id).first()
        if not user:
            return None

        user.activate_subscription(days)
        db.commit()
        db.refresh(user)
        return user
    finally:
        db.close()


def grant_user_subscription(telegram_id: int, username: str | None, days: float) -> User:
    db = get_db()
    try:
        user = db.query(User).filter(User.telegram_id == telegram_id).first()
        if not user:
            user = User(telegram_id=telegram_id, username=username)
            db.add(user)
            db.flush()

        user.activate_subscription(days)
        db.commit()
        db.refresh(user)
        return user
    finally:
        db.close()


def create_invoice_record(telegram_id: int, invoice_id: int, plan_type: str, amount: float, payload: str) -> None:
    db = get_db()
    try:
        existing = db.query(PaymentInvoice).filter(PaymentInvoice.invoice_id == invoice_id).first()
        if existing:
            return

        invoice = PaymentInvoice(
            invoice_id=invoice_id,
            telegram_id=telegram_id,
            plan_type=plan_type,
            amount=amount,
            payload=payload,
            status="pending",
        )
        db.add(invoice)
        db.commit()
    finally:
        db.close()


def mark_invoice_paid(invoice_id: int, raw_update: dict) -> bool:
    db = get_db()
    try:
        invoice = db.query(PaymentInvoice).filter(PaymentInvoice.invoice_id == invoice_id).first()
        if not invoice:
            return False

        if invoice.status == "paid":
            return False

        invoice.status = "paid"
        invoice.paid_at = datetime.utcnow()
        invoice.raw_update = raw_update
        db.commit()
        return True
    finally:
        db.close()


def get_invoice(invoice_id: int) -> Optional[PaymentInvoice]:
    db = get_db()
    try:
        return db.query(PaymentInvoice).filter(PaymentInvoice.invoice_id == invoice_id).first()
    finally:
        db.close()


def get_admin_stats() -> dict:
    db = get_db()
    try:
        now = datetime.utcnow()
        total_users = db.query(User).count()
        active_subscriptions = db.query(User).filter(User.subscription_until.is_not(None), User.subscription_until > now).count()
        active_demo = db.query(User).filter(User.subscription_until.is_not(None), User.subscription_until > now, User.is_demo.is_(True)).count()
        paid_active = db.query(User).filter(User.subscription_until.is_not(None), User.subscription_until > now, User.is_demo.is_(False)).count()
        paid_invoices = db.query(PaymentInvoice).filter(PaymentInvoice.status == "paid").count()
        pending_invoices = db.query(PaymentInvoice).filter(PaymentInvoice.status == "pending").count()
        pending_manual_requests = db.query(ManualPaymentRequest).filter(ManualPaymentRequest.status == "pending").count()
        return {
            "total_users": total_users,
            "active_subscriptions": active_subscriptions,
            "active_demo": active_demo,
            "paid_active": paid_active,
            "paid_invoices": paid_invoices,
            "pending_invoices": pending_invoices,
            "pending_manual_requests": pending_manual_requests,
        }
    finally:
        db.close()


def create_manual_payment_request(telegram_id: int, username: str | None, plan_type: str, amount: float) -> ManualPaymentRequest:
    db = get_db()
    try:
        req = ManualPaymentRequest(
            telegram_id=telegram_id,
            username=username,
            plan_type=plan_type,
            amount=amount,
            status="pending",
        )
        db.add(req)
        db.commit()
        db.refresh(req)
        return req
    finally:
        db.close()


def get_manual_payment_request(request_id: int) -> Optional[ManualPaymentRequest]:
    db = get_db()
    try:
        return db.query(ManualPaymentRequest).filter(ManualPaymentRequest.id == request_id).first()
    finally:
        db.close()


def list_pending_manual_payment_requests(limit: int = 20) -> list[ManualPaymentRequest]:
    db = get_db()
    try:
        return (
            db.query(ManualPaymentRequest)
            .filter(ManualPaymentRequest.status == "pending")
            .order_by(ManualPaymentRequest.created_at.asc())
            .limit(limit)
            .all()
        )
    finally:
        db.close()


def set_manual_payment_status(request_id: int, status: str, reviewed_by: int, comment: str | None = None) -> Optional[ManualPaymentRequest]:
    db = get_db()
    try:
        req = db.query(ManualPaymentRequest).filter(ManualPaymentRequest.id == request_id).first()
        if not req:
            return None
        if req.status != "pending":
            return req

        req.status = status
        req.reviewed_by = reviewed_by
        req.reviewed_at = datetime.utcnow()
        if comment:
            req.comment = comment
        db.commit()
        db.refresh(req)
        return req
    finally:
        db.close()
