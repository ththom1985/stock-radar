"""Quarter-over-quarter LLM tone shift from genuine earnings-call transcripts."""
from __future__ import annotations
import json, os
from datetime import datetime, timedelta, timezone
from .config import DATA
from .persistence import atomic_write_json, cache_failure, clear_cache_failure, load_json, schema_meta, utc_now

CACHE_PATH = DATA / "earnings_tone.json"
MAX_AGE_DAYS = 30
MODEL = os.environ.get("STOCK_RADAR_LLM_MODEL", "claude-haiku-4-5")
SCHEMA = {"type":"object","properties":{"current_tone":{"type":"integer"},"previous_tone":{"type":"integer"},"hedging_shift":{"type":"integer"},"qa_evasiveness_shift":{"type":"integer"},"cfo_tone_shift":{"type":"integer"},"reason":{"type":"string"}},"required":["current_tone","previous_tone","hedging_shift","qa_evasiveness_shift","cfo_tone_shift","reason"],"additionalProperties":False}

def _excerpt(text):
    text = " ".join(str(text).split())
    return text[:12000] + "\n[...]\n" + text[-12000:] if len(text)>24000 else text

def _llm_compare(symbol, current, previous):
    import anthropic
    client=anthropic.Anthropic()
    system=("Compare two genuine quarterly earnings-call transcripts. Score tone 0-100; "
            "positive shifts are positive. Detect hedging/challenging/headwinds language and "
            "evasive analyst-Q&A answers. Weight CFO statements 1.5x CEO statements. Return "
            "only the requested JSON. Do not infer facts absent from transcripts.")
    user=f"{symbol} CURRENT:\n{_excerpt(current)}\n\nPREVIOUS:\n{_excerpt(previous)}"
    response=client.messages.create(model=MODEL,max_tokens=900,system=system,messages=[{"role":"user","content":user}],output_config={"format":{"type":"json_schema","schema":SCHEMA}})
    text=next((block.text for block in response.content if block.type=="text"),"{}")
    return json.loads(text)

def build_tone_signal(result, current_event, previous_event):
    current=max(0,min(100,int(result["current_tone"]))); previous=max(0,min(100,int(result["previous_tone"])))
    tone_shift=current-previous
    composite=tone_shift-int(result["hedging_shift"])-int(result["qa_evasiveness_shift"])+1.5*int(result["cfo_tone_shift"])
    score=round(max(0,min(100,50+composite)),1)
    return {"score":score,"direction":"positive" if score>55 else "negative" if score<45 else "neutral","current_tone":current,"previous_tone":previous,"tone_shift":tone_shift,"hedging_shift":int(result["hedging_shift"]),"qa_evasiveness_shift":int(result["qa_evasiveness_shift"]),"cfo_tone_shift":int(result["cfo_tone_shift"]),"reason":result["reason"],"current_period":current_event,"previous_period":previous_event,"source":"earningscall.biz genuine call transcripts + Claude comparison","expected_delay":"transcript availability after earnings call","cfo_weight":1.5}

def _fresh(entry):
    try:
        value=datetime.fromisoformat(str((entry or {}).get("last_success_at"))); value=value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    except (TypeError,ValueError): return False
    return value>=datetime.now(timezone.utc)-timedelta(days=MAX_AGE_DAYS)

def fetch_earnings_tone_signals(rows,max_new=None,force=False,verbose=True):
    cache=load_json(CACHE_PATH,expected_type=dict,default={})
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return {},{"status":"disabled","reason":"ANTHROPIC_API_KEY unavailable","refreshed":0}
    if not (
        os.environ.get("EARNINGSCALL_API_KEY")
        or os.environ.get("ECALL_API_KEY")
    ):
        return {},{
            "status":"disabled",
            "reason":"EARNINGSCALL_API_KEY unavailable",
            "registration":"https://earningscall.biz/api-pricing",
            "refreshed":0,
        }
    eligible=[row for row in rows if row.get("asset_type")=="company_equity" and row.get("currency")=="USD"]
    eligible.sort(key=lambda row:(-(row.get("longterm_score") or 0),row["symbol"]))
    max_new=max_new if max_new is not None else int(os.environ.get("STOCK_RADAR_EARNINGS_TONE_MAX","3"))
    candidates=[row for row in eligible if force or not _fresh(cache.get(row["symbol"]))][:max_new]
    failures={}
    import earningscall
    for index,row in enumerate(candidates,1):
        symbol=row["symbol"]
        try:
            company=earningscall.get_company(symbol); events=[event for event in company.events() if event.conference_date and event.conference_date.timestamp()<=datetime.now(timezone.utc).timestamp()]
            if len(events)<2: raise ValueError("fewer than two completed earnings calls")
            transcripts=[company.get_transcript(event=event) for event in events[:2]]
            if not all(transcripts) or not all(len(item.text or "")>3000 for item in transcripts): raise ValueError("genuine transcript unavailable")
            result=_llm_compare(symbol,transcripts[0].text,transcripts[1].text)
            signal=build_tone_signal(result,{"year":events[0].year,"quarter":events[0].quarter,"conference_date":events[0].conference_date.isoformat()},{"year":events[1].year,"quarter":events[1].quarter,"conference_date":events[1].conference_date.isoformat()})
            signal.update({"fetched_at":utc_now(),"last_success_at":utc_now()}); cache[symbol]=clear_cache_failure(signal)
        except Exception as exc:
            cache[symbol]=cache_failure(cache.get(symbol),exc); failures[symbol]=str(exc)[:200]
        if verbose: print(f"  Earnings tone {index}/{len(candidates)}")
    if candidates:
        cache["_meta"]=schema_meta("stock-radar-earnings-tone-cache",schema_version=1,refreshed=len(candidates)); atomic_write_json(CACHE_PATH,cache,indent=1)
    return {row["symbol"]:cache.get(row["symbol"]) for row in rows if isinstance(cache.get(row.get("symbol")),dict) and cache.get(row["symbol"],{}).get("score") is not None},{"status":"ok" if not failures else "partial","eligible":len(eligible),"cached":sum(_fresh(cache.get(row["symbol"])) for row in eligible),"refreshed":len(candidates),"signal_count":sum((cache.get(row["symbol"]) or {}).get("score") is not None for row in eligible),"failures":failures,"model":MODEL}
