"""Durable single-controller reservations; not a provider billing control.

Use a local SQLite file owned by one trusted controller. Do not put this database
on a Modal Volume shared by multiple live writers. Every paid adapter must use
the same controller; direct provider calls are outside this guard's coverage.
"""
from .diagnostics import diagnosed
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import sqlite3
from .analysis_policy import PROVIDER_CAPS_USD
from .diagnostics import diagnostic_issue


def units(value):
    if isinstance(value, bool):
        raise ValueError("A finite nonnegative USD amount is required")
    try:
        amount = Decimal(str(value))
        if not amount.is_finite() or amount < 0 or amount > Decimal("1000000"):
            raise ValueError("A finite bounded nonnegative USD amount is required")
        return int((amount * 1000000).to_integral_value(rounding=ROUND_CEILING))
    except (InvalidOperation, TypeError) as exc:
        raise ValueError("Invalid USD amount") from exc


class SpendingLedger:
    """Atomic reservations, idempotency and fail-closed reconciliation.

    Opening a ledger never enables any pool. Operator verification must supply
    a fixed baseline of prior charges and *unreserved* eligible credit headroom.
    Rates, bounded units/retries/runtime and cancellation remain caller duties.
    """
    def __init__(self, path):
        self.path = str(path)
        with self._transaction() as db:
            db.execute("CREATE TABLE IF NOT EXISTS pools (provider TEXT PRIMARY KEY, prior INTEGER NOT NULL, headroom INTEGER NOT NULL, verified INTEGER NOT NULL DEFAULT 0, blocked INTEGER NOT NULL DEFAULT 0)")
            db.execute("CREATE TABLE IF NOT EXISTS reservations (id TEXT PRIMARY KEY, provider TEXT NOT NULL, fingerprint TEXT NOT NULL, maximum INTEGER NOT NULL, charge INTEGER, state TEXT NOT NULL)")
            for provider in PROVIDER_CAPS_USD:
                db.execute("INSERT OR IGNORE INTO pools VALUES (?,0,0,0,0)", (provider,))

    @contextmanager
    def _transaction(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    @diagnosed("budget.verify")
    def verify_pool(self, provider, *, prior_spend_usd, available_credit_usd):
        if provider not in PROVIDER_CAPS_USD:
            raise ValueError("Unknown provider")
        prior, headroom = units(prior_spend_usd), units(available_credit_usd)
        with self._transaction() as db:
            old = db.execute("SELECT verified, prior FROM pools WHERE provider=?", (provider,)).fetchone()
            count = db.execute("SELECT COUNT(*) FROM reservations WHERE provider=?", (provider,)).fetchone()[0]
            if old[0] or count:
                raise ValueError("Baseline already fixed; reconcile instead of resetting budget")
            db.execute("UPDATE pools SET prior=?,headroom=?,verified=1 WHERE provider=?", (prior,headroom,provider))

    def block_pool(self,provider):
        if provider not in PROVIDER_CAPS_USD:raise ValueError('Unknown provider')
        with self._transaction() as db:
            db.execute('UPDATE pools SET blocked=1 WHERE provider=?',(provider,))

    def can_reserve(self, provider, worst_case_usd):
        """Read-only readiness check; reserve() remains the atomic execution gate."""
        if provider not in PROVIDER_CAPS_USD:
            return False
        maximum = units(worst_case_usd)
        with sqlite3.connect(self.path) as db:
            row = db.execute(
                "SELECT prior,headroom,verified,blocked FROM pools WHERE provider=?", (provider,)
            ).fetchone()
            if row is None or not row[2] or row[3]:
                return False
            used = db.execute(
                "SELECT COALESCE(SUM(CASE WHEN state='final' THEN charge ELSE maximum END),0) "
                "FROM reservations WHERE provider=?", (provider,)
            ).fetchone()[0]
            return maximum <= min(units(PROVIDER_CAPS_USD[provider]) - row[0], row[1]) - used

    @diagnosed("budget.reserve")
    def reserve(self, provider, request_id, *, fingerprint, worst_case_usd):
        if provider not in PROVIDER_CAPS_USD or not request_id or not fingerprint:
            raise ValueError("Known provider, request ID and immutable request fingerprint required")
        maximum = units(worst_case_usd)
        if maximum <= 0:
            raise ValueError("Positive bounded cost including every attempt is required")
        with self._transaction() as db:
            old = db.execute("SELECT provider,fingerprint,maximum,state FROM reservations WHERE id=?",(request_id,)).fetchone()
            if old:
                if old[:3] != (provider,fingerprint,maximum):
                    raise ValueError("Request ID reused with changed inputs or cost")
                # Caller must return existing job/result; this never grants a retry.
                return {"request_id":request_id,"created":False,"state":old[3]}
            prior,headroom,verified,blocked = db.execute("SELECT prior,headroom,verified,blocked FROM pools WHERE provider=?",(provider,)).fetchone()
            if not verified or blocked:
                raise ValueError("Pool is unverified or blocked; no paid work allowed")
            used = db.execute("SELECT COALESCE(SUM(CASE WHEN state='final' THEN charge ELSE maximum END),0) FROM reservations WHERE provider=?",(provider,)).fetchone()[0]
            remaining = min(units(PROVIDER_CAPS_USD[provider])-prior,headroom)-used
            if maximum > remaining:
                raise ValueError("Reservation exceeds remaining 50-percent project/credit ceiling")
            db.execute("INSERT INTO reservations VALUES (?,?,?,?,NULL,'reserved')",(request_id,provider,fingerprint,maximum))
            return {"request_id":request_id,"created":True,"state":"reserved"}

    @diagnosed("budget.reconcile")
    def reconcile(self, request_id, *, actual_usd=None, finalized=False):
        if type(finalized) is not bool:
            raise ValueError("Finalized must be explicit boolean")
        actual = units(actual_usd) if actual_usd is not None else None
        if finalized and actual is None:
            raise ValueError("Finalized provider charge is required")
        overrun=False
        with self._transaction() as db:
            row=db.execute("SELECT provider,maximum,charge,state FROM reservations WHERE id=?",(request_id,)).fetchone()
            if row is None:
                raise ValueError("Unknown request")
            provider,maximum,previous,state=row
            if state=='final':
                if finalized and actual==previous:return
                raise ValueError("Final charge is immutable; manual audit required")
            if actual is not None and actual > maximum:
                # Persist the observed breach and block further launches; never
                # roll back the evidence by throwing inside this transaction.
                db.execute("UPDATE pools SET blocked=1 WHERE provider=?",(provider,))
                db.execute("UPDATE reservations SET charge=?,state='overrun' WHERE id=?",(actual,request_id))
                overrun=True
            elif state=='overrun':
                raise ValueError("Overrun requires manual audit; cannot clear through reconciliation")
            else:
                db.execute("UPDATE reservations SET charge=?,state=? WHERE id=?",(actual,'final' if finalized else 'pending',request_id))
        if overrun:
            diagnostic_issue('budget.reconcile','BUDGET_OVERRUN')
