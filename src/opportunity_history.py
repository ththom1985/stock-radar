"""Rolling deal-quality reference distribution for decision-facing views."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .config import DATA
from .persistence import atomic_write_json, load_json, schema_meta

HISTORY_PATH = DATA / "opportunity_history.json"
RETENTION_DAYS = 365
MIN_RELIABLE_OBSERVATIONS = 100
MIN_RELIABLE_CALENDAR_DAYS = 30


def update_opportunity_history(question_views, observed_at=None):
    observed_at = observed_at or datetime.now(timezone.utc)
    cutoff = (observed_at - timedelta(days=RETENTION_DAYS)).date().isoformat()
    payload = load_json(HISTORY_PATH, expected_type=dict, default={})
    snapshots = [
        snapshot
        for snapshot in payload.get("snapshots") or []
        if isinstance(snapshot, dict)
        and str(snapshot.get("date") or "") >= cutoff
    ]
    current = {
        "date": observed_at.date().isoformat(),
        "scores": [
            item["deal_quality"]["score"]
            for item in question_views.get("cheap_with_potential") or []
            if isinstance((item.get("deal_quality") or {}).get("score"), (int, float))
        ],
    }
    snapshots = [
        snapshot
        for snapshot in snapshots
        if snapshot.get("date") != current["date"]
    ]
    snapshots.append(current)
    snapshots.sort(key=lambda snapshot: snapshot["date"])
    atomic_write_json(
        HISTORY_PATH,
        {
            "schema": "stock-radar-opportunity-history",
            "schema_version": 1,
            "retention_days": RETENTION_DAYS,
            "snapshots": snapshots,
            "_meta": schema_meta(
                "stock-radar-opportunity-history",
                schema_version=1,
                retention_days=RETENTION_DAYS,
            ),
        },
        indent=1,
    )
    calendar_days = (
        (
            datetime.fromisoformat(snapshots[-1]["date"]).date()
            - datetime.fromisoformat(snapshots[0]["date"]).date()
        ).days
        + 1
        if snapshots
        else 0
    )
    scores = [
        score
        for snapshot in snapshots
        for score in snapshot.get("scores") or []
        if isinstance(score, (int, float))
    ]
    return {
        "scores": scores,
        "snapshot_count": len(snapshots),
        "calendar_days": calendar_days,
        "from_date": snapshots[0]["date"] if snapshots else None,
        "to_date": snapshots[-1]["date"] if snapshots else None,
        "reliable": (
            len(scores) >= MIN_RELIABLE_OBSERVATIONS
            and calendar_days >= MIN_RELIABLE_CALENDAR_DAYS
        ),
        "reliability_requirement": (
            f"mindestens {MIN_RELIABLE_OBSERVATIONS} Gelegenheiten über "
            f"mindestens {MIN_RELIABLE_CALENDAR_DAYS} Kalendertage"
        ),
    }
