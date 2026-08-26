"""Plain-German presentation layer derived only from existing row fields."""
from __future__ import annotations
import math

STATUS_LABELS={
    "in_zone_confirmed":"In der Kaufzone ✓",
    "in_zone_risk_filtered":"Zone erreicht, Risiko bremst",
    "approaching":"Fast in der Kaufzone",
    "setup_waiting_confirmation":"Bestätigung abwarten",
    "safety_blocked":"Kein Einstieg",
    "broken_below":"Technisches Bild gebrochen",
    "far_above":"Zu weit über der Zone",
    "reference_only_far":"Nur beobachten",
    "unavailable":"Keine belastbare Zone",
}
VERDICTS={"clearly_undervalued":"deutlich unterbewertet","fair":"fair bewertet","expensive":"eher teuer","overpriced":"deutlich überteuert","unavailable":"nicht belastbar bewertbar","data_review_required":"derzeit nicht belastbar bewertet"}
GROUP_LABELS={"insider":"Insider-Aktivität","congress":"Congress-Transaktionen","attention":"steigende Aufmerksamkeit (Wikipedia-Aufrufe oder Jobs)","earnings_tone":"positiver Tonwechsel in den Quartalsunterlagen"}

def _num(value):
    return float(value) if isinstance(value,(int,float)) and math.isfinite(value) else None

def _money(value,currency):
    if value is None:return "–"
    symbols={"USD":"$","EUR":"€","GBP":"£","JPY":"¥","TWD":"TWD","HKD":"HKD"}
    digits=2 if abs(value)>=1 else 4
    rendered=f"{value:,.{digits}f}".replace(",","X").replace(".",",").replace("X",".")
    return f"{rendered} {symbols.get(currency,currency or '')}".strip()

def local_zone(row):
    sweet=row.get("sweet_spot") or {}; rate=_num(row.get("fx_usd"))
    values=[_num(sweet.get(key)) for key in ("lower","ideal","upper")]
    if not rate or not all(value is not None for value in values):return None
    return {"lower":values[0]/rate,"ideal":values[1]/rate,"upper":values[2]/rate,"currency":row.get("currency")}

def traffic_light(row):
    status=(row.get("sweet_spot") or {}).get("combined_status")
    if status=="in_zone_confirmed":return "green"
    if status in {"in_zone_confirmed","approaching","setup_waiting_confirmation","in_zone_risk_filtered"}:return "yellow"
    return "red"

def plain_language(row):
    expert=row.get("expert_analysis") or {}; valuation=expert.get("valuation") or {}; fair=valuation.get("fair_value_range") or {}
    currency=row.get("currency"); price=_num(row.get("price_local")); verdict=valuation.get("verdict","unavailable")
    fair_text=(f"der faire Bereich liegt bei {_money(fair.get('lower'),currency)} bis {_money(fair.get('upper'),currency)}" if fair else "ein fairer Bereich ist noch nicht belastbar")
    if verdict=="data_review_required":
        valuation_text=f"Die Bewertung von {row.get('display_name_full') or row.get('name') or row.get('symbol')} ist derzeit nicht belastbar; Kurs {_money(price,currency)}."
    else:
        valuation_text=f"{row.get('display_name_full') or row.get('name') or row.get('symbol')} ist aktuell {VERDICTS.get(verdict,verdict)} — {fair_text}, der Kurs bei {_money(price,currency)}."
    zone=local_zone(row); status=(row.get("sweet_spot") or {}).get("combined_status")
    if zone and status=="in_zone_confirmed":
        timing=f"Der Kurs liegt in der technischen Einstiegszone von {_money(zone['lower'],currency)} bis {_money(zone['upper'],currency)}."
    elif zone:
        timing=f"Interessanter wird der Kurs in der Zone von {_money(zone['lower'],currency)} bis {_money(zone['upper'],currency)}."
    else: timing="Eine belastbare Einstiegszone fehlt derzeit."
    alt=row.get("alternative_signals") or {}; groups=alt.get("contributing_groups") or []
    positive=[GROUP_LABELS.get(item["group"],item["group"]) for item in groups if _num(item.get("score")) is not None and item["score"]>55]
    signal=("Dafür spricht: "+", ".join(positive)+"." if positive else "Derzeit gibt es keine klar positiven unabhängigen Signalgruppen.")
    if verdict in {"expensive","overpriced"}:signal+=" Dagegen spricht die hohe Bewertung."
    risks=(expert.get("risks") or {}).get("top_risks") or row.get("risk_warnings") or []
    if risks and str(risks[0]).startswith("Das Modell erkennt aktuell kein"):
        risk=risks[0]
    else:
        risk=f"Größtes sichtbares Risiko: {risks[0]}" if risks else "Das Modell erkennt aktuell kein einzelnes dominantes Risiko."
    return {"valuation":valuation_text,"timing":timing,"signals":signal,"risk":risk}

def _reason(row,light):
    verdict=((row.get("expert_analysis") or {}).get("valuation") or {}).get("verdict")
    alt=row.get("alternative_signals") or {}; groups=alt.get("contributing_groups") or []
    if verdict in {"data_review_required","unavailable"}:
        count=len(groups); group_text=f"{count} unabhängige Signalgruppe" if count==1 else f"{count} unabhängige Signalgruppen"
        return "Kurs in der technischen Unterstützungszone"+(f", {group_text} vorhanden." if groups else ".")+" Die Bewertung wird nicht verwendet."
    if light=="green":
        count=len(groups); group_text=f"{count} unabhängige Signalgruppe" if count==1 else f"{count} unabhängige Signalgruppen"
        return f"{VERDICTS.get(verdict,'Fair bewertet').capitalize()}, Kurs in der Unterstützungszone"+(f", {group_text} vorhanden." if groups else ".")
    if light=="yellow" and verdict in {"expensive","overpriced"}:return "Technische Zone erreicht, aber die Bewertung ist hoch."
    if light=="yellow":return "Der Kurs nähert sich einer interessanten Zone; Bestätigung abwarten."
    return "Timing, Bewertung oder Sicherheitsfilter sprechen derzeit gegen einen Einstieg."

def build_today_view(rows,previous_snapshot=None,price_histories=None,limit=5):
    cards=[]
    for row in rows:
        if row.get("asset_type")!="company_equity":continue
        sweet=row.get("sweet_spot") or {}; light=traffic_light(row); zone=local_zone(row)
        card={"symbol":row.get("symbol"),"name":row.get("display_name_full") or row.get("name"),"currency":row.get("currency"),"price":row.get("price_local"),"light":light,"status":sweet.get("combined_status"),"status_label":STATUS_LABELS.get(sweet.get("combined_status"),"Beobachten"),"zone":zone,"why":_reason(row,light),"plain":plain_language(row),"earnings_in_days":row.get("earnings_in_days"),"confluence_tier":(row.get("alternative_signals") or {}).get("confluence_tier"),"contributing_groups":(row.get("alternative_signals") or {}).get("contributing_groups") or [],"timing_score":row.get("entry_timing_score"),"evidence_score":sweet.get("reliability_score")}
        if price_histories and row.get("symbol") in price_histories:
            frame=price_histories[row["symbol"]]
            card["sparkline"]=[round(float(value),6) for value in frame["RawClose"].dropna().tail(30)]
        cards.append(card)
    order={"green":0,"yellow":1,"red":2}
    cards.sort(key=lambda item:(order[item["light"]],-(item.get("evidence_score") or 0),-(item.get("timing_score") or 0),item["symbol"]))
    green=[item for item in cards if item["light"]=="green"]
    selected=(green[:limit] if green else cards[:1])
    if green:
        headline=f"{len(green)} Titel liegen heute in einer überzeugenden Einstiegszone"
        summary="Gezeigt werden die stärksten Hinweise aus Timing und Bewertung."
    else:
        best=cards[0] if cards else None
        headline="Heute kein überzeugender Einstieg"
        summary=(f"Bester Beobachtungskandidat: {best['name']}." if best else "Es liegen keine belastbaren Kandidaten vor.")
    previous_by_symbol={row.get("symbol"):row for row in (previous_snapshot or {}).get("all",[])}
    changes=[]
    for item in cards:
        previous=previous_by_symbol.get(item["symbol"]) or {}; old=(previous.get("sweet_spot") or {}).get("combined_status")
        if old and old!=item["status"]:
            changes.append(f"{item['symbol']}: {STATUS_LABELS.get(old,old)} → {item['status_label']}")
    return {"headline":headline,"summary":summary,"candidate_count":len(green),"candidates":selected,"changes":changes[:6],"disclaimer":"Hinweise nach Systemlogik, keine Anlageberatung. Die Entscheidung bleibt beim Nutzer."}
