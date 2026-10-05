"""Alert/feedback persistence (SQLAlchemy: SQLite by default, PostgreSQL via DATABASE_URL)
plus an optional Redis cache."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine, select, update
from sqlalchemy.orm import Session, declarative_base

Base = declarative_base()


class Alert(Base):
    __tablename__ = "alerts"
    id = Column(Integer, primary_key=True)
    flow_id = Column(Integer, index=True)
    created = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    src_ip = Column(String(64), index=True)
    dst_ip = Column(String(64))
    risk = Column(Float)
    verdict = Column(String(80))
    pred_class = Column(String(40))
    features = Column(Text)   # JSON feature vector, used for feedback-driven updates


class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True)
    flow_id = Column(Integer, index=True)
    label = Column(String(40))
    analyst = Column(String(80))
    created = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    used = Column(Integer, default=0)


class Store:
    def __init__(self, url: str | None = None):
        url = url or os.environ.get("DATABASE_URL", "sqlite:///argus.db")
        self.engine = create_engine(url, future=True)
        Base.metadata.create_all(self.engine)

    def add_alerts(self, flows: pd.DataFrame, X: np.ndarray) -> int:
        m = flows["is_anomaly"].to_numpy()
        rows = [Alert(flow_id=int(r.flow_id), src_ip=r.src_ip, dst_ip=r.dst_ip, risk=float(r.risk),
                      verdict=r.verdict, pred_class=r.pred_class,
                      features=json.dumps(X[i].tolist()))
                for i, r in zip(np.where(m)[0], flows[m].itertuples())]
        with Session(self.engine) as s:
            s.add_all(rows); s.commit()
        return len(rows)

    def alerts(self, min_risk: float = 0.0, limit: int = 100) -> pd.DataFrame:
        q = select(Alert.flow_id, Alert.created, Alert.src_ip, Alert.dst_ip, Alert.risk,
                   Alert.verdict, Alert.pred_class).where(Alert.risk >= min_risk) \
            .order_by(Alert.risk.desc()).limit(limit)
        return pd.read_sql(q, self.engine)

    def add_feedback(self, flow_id: int, label: str, analyst: str) -> None:
        with Session(self.engine) as s:
            s.add(Feedback(flow_id=flow_id, label=label, analyst=analyst)); s.commit()

    def pending_feedback(self) -> tuple[np.ndarray, np.ndarray, list[int]]:
        with Session(self.engine) as s:
            fb = s.execute(select(Feedback).where(Feedback.used == 0)).scalars().all()
            X, y, ids = [], [], []
            for f in fb:
                a = s.execute(select(Alert).where(Alert.flow_id == f.flow_id)
                              .order_by(Alert.id.desc())).scalars().first()
                if a is not None:
                    X.append(json.loads(a.features)); y.append(f.label); ids.append(f.id)
            return np.array(X, dtype=float), np.array(y), ids

    def mark_used(self, ids: list[int]) -> None:
        with Session(self.engine) as s:
            s.execute(update(Feedback).where(Feedback.id.in_(ids)).values(used=1)); s.commit()


class Cache:
    """Redis if REDIS_URL is set and reachable; otherwise a no-op."""

    def __init__(self):
        self.r = None
        url = os.environ.get("REDIS_URL")
        if url:
            try:
                import redis
                self.r = redis.Redis.from_url(url, socket_timeout=1)
                self.r.ping()
            except Exception:
                self.r = None

    def get(self, key: str):
        try:
            v = self.r.get(key) if self.r else None
            return json.loads(v) if v else None
        except Exception:
            return None

    def set(self, key: str, value, ttl: int = 15) -> None:
        try:
            if self.r:
                self.r.setex(key, ttl, json.dumps(value, default=float))
        except Exception:
            pass

    def clear(self, key: str) -> None:
        try:
            if self.r:
                self.r.delete(key)
        except Exception:
            pass
