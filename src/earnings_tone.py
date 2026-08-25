"""Free earnings-language shifts from EDGAR 8-K exhibits and configured IR pages."""
from __future__ import annotations
import html, json, os, re, time
from datetime import datetime, timedelta, timezone
from .config import DATA
from .persistence import atomic_write_json, cache_failure, clear_cache_failure, load_json, schema_meta, utc_now
from .sec_companyfacts import load_sec_ticker_map, normalize_sec_user_agent, request_sec_json
from .sec_insiders import _request_text

CACHE_PATH=DATA/"earnings_tone.json"
SOURCE_CACHE_PATH=DATA/"earnings_text_sources.json"
IR_CONFIG_PATH=DATA/"ir_sources.json"
SUBMISSIONS_URL="https://data.sec.gov/submissions/CIK{cik:010d}.json"
INDEX_URL="https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/index.json"
ARCHIVE_URL="https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
MAX_AGE_DAYS=30
MODEL="free_financial_lexicon_v1"
POSITIVE={"strong","growth","record","improved","increase","opportunity","resilient","momentum","profit","confident","demand"}
NEGATIVE={"decline","weak","loss","adverse","constraint","shortage","litigation","impairment","risk","pressure","decrease"}
HEDGING={"may","might","could","uncertain","approximately","expect","believe","anticipate","subject","potentially","challenging","headwind"}
KEYWORDS=("quarter","revenue","earnings","results","net income","guidance","outlook")

def _plain(value):
    value=re.sub(r"(?is)<script.*?</script>|<style.*?</style>"," ",value)
    return " ".join(html.unescape(re.sub(r"(?s)<[^>]+>"," ",value)).split())

def _recent_8k(submissions):
    recent=(submissions.get("filings") or {}).get("recent") or {}; result=[]
    for i,form in enumerate(recent.get("form") or []):
        if form!="8-K": continue
        result.append({"filing_date":recent["filingDate"][i],"report_date":recent["reportDate"][i],"accession":recent["accessionNumber"][i],"primary_document":recent["primaryDocument"][i].split("/")[-1]})
        if len(result)>=12: break
    return result

def _candidate_documents(index_payload,primary):
    items=((index_payload.get("directory") or {}).get("item") or []); ranked=[]
    for item in items:
        name=str(item.get("name") or ""); lower=name.casefold()
        if name==primary or not lower.endswith((".htm",".html",".txt")): continue
        size=int(item.get("size") or 0); score=size
        if "ex99" in lower or "ex-99" in lower: score+=10_000_000
        ranked.append((score,name))
    return [name for _,name in sorted(ranked,reverse=True)[:4]]

def fetch_edgar_prepared_texts(symbol,ua):
    cik=load_sec_ticker_map(ua).get(symbol.upper())
    if not cik: return []
    filings=_recent_8k(request_sec_json(SUBMISSIONS_URL.format(cik=cik),ua)); found=[]
    for filing in filings:
        accession=filing["accession"].replace("-","")
        index=request_sec_json(INDEX_URL.format(cik=cik,accession=accession),ua)
        for document in _candidate_documents(index,filing["primary_document"]):
            url=ARCHIVE_URL.format(cik=cik,accession=accession,document=document)
            text=_plain(_request_text(url,ua))
            lower=text.casefold()
            if len(text)>=2500 and sum(keyword in lower for keyword in KEYWORDS)>=4:
                full=bool(re.search(r"(?i)question[- ]and[- ]answer|questions?\s+(?:from|and)\s+analysts",text))
                found.append({"period":filing["report_date"],"filing_date":filing["filing_date"],"text":text,"status":"full" if full else "prepared-only","source":"SEC EDGAR 8-K exhibit","source_url":url})
                break
        if len(found)>=2: break
        time.sleep(.15)
    return found

def _excerpt(text):
    text=" ".join(text.split()); return text[:11000]+("\n[...]\n"+text[-11000:] if len(text)>22000 else "")

def _lexicon_metrics(text):
    words=re.findall(r"[a-z]+",text.casefold()); total=max(1,len(words))
    positive=sum(word in POSITIVE for word in words); negative=sum(word in NEGATIVE for word in words); hedging=sum(word in HEDGING for word in words)
    tone=round(max(0,min(100,50+(positive-negative)/total*2500)))
    return {"tone":tone,"hedging_rate":hedging/total*1000,"positive":positive,"negative":negative,"words":total}

def _compare(current,previous):
    now=_lexicon_metrics(current["text"]); before=_lexicon_metrics(previous["text"])
    return {"current_tone":now["tone"],"previous_tone":before["tone"],"hedging_shift":round(now["hedging_rate"]-before["hedging_rate"]),"qa_evasiveness_shift":None,"cfo_tone_shift":None,"reason":f"Free lexicon: positive/negative {now['positive']}/{now['negative']} vs {before['positive']}/{before['negative']}; Q&A/CFO unavailable."}

def build_tone_signal(result,current,previous):
    now=max(0,min(100,int(result["current_tone"]))); before=max(0,min(100,int(result["previous_tone"]))); shift=now-before
    qa=result.get("qa_evasiveness_shift"); cfo=result.get("cfo_tone_shift")
    parts=[shift-int(result["hedging_shift"])]; weights=[1.0]
    if qa is not None: parts.append(-int(qa)); weights.append(1.0)
    if cfo is not None: parts.append(1.5*int(cfo)); weights.append(1.5)
    composite=sum(parts)/sum(weights); score=round(max(0,min(100,50+composite)),1)
    return {"score":score,"direction":"positive" if score>55 else "negative" if score<45 else "neutral","transcript_status":"full" if current["status"]=="full" and previous["status"]=="full" else "prepared-only","analysis_method":MODEL,"current_tone":now,"previous_tone":before,"tone_shift":shift,"hedging_shift":int(result["hedging_shift"]),"qa_evasiveness_shift":qa,"qa_status":"available" if qa is not None else "not_available","cfo_tone_shift":cfo,"cfo_weight":1.5 if cfo is not None else None,"reason":result["reason"],"current_period":{k:current[k] for k in ("period","filing_date","source","source_url")},"previous_period":{k:previous[k] for k in ("period","filing_date","source","source_url")},"source":"free earnings text + transparent financial lexicon","expected_delay":"source publication time"}

def _fresh(entry):
    try:
        value=datetime.fromisoformat(str((entry or {}).get("last_success_at"))); value=value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    except (TypeError,ValueError): return False
    return value>=datetime.now(timezone.utc)-timedelta(days=MAX_AGE_DAYS)

def fetch_earnings_tone_signals(rows,max_new=None,force=False,verbose=True):
    cache=load_json(CACHE_PATH,expected_type=dict,default={}); sources=load_json(SOURCE_CACHE_PATH,expected_type=dict,default={})
    ua=normalize_sec_user_agent()
    if not ua: return {},{**coverage_status(rows,sources,cache),"status":"disabled","reason":"SEC_USER_AGENT unavailable"}
    eligible=[row for row in rows if row.get("asset_type")=="company_equity" and row.get("currency")=="USD"]; eligible.sort(key=lambda row:(-(row.get("longterm_score") or 0),row["symbol"]))
    max_new=max_new if max_new is not None else int(os.environ.get("STOCK_RADAR_EARNINGS_TONE_MAX","3")); candidates=[row for row in eligible if force or not _fresh(cache.get(row["symbol"]))][:max_new]; failures={}
    for index,row in enumerate(candidates,1):
        symbol=row["symbol"]
        try:
            texts=fetch_edgar_prepared_texts(symbol,ua); sources[symbol]={"status":texts[0]["status"] if texts else "none","period_count":len(texts),"updated_at":utc_now()}
            if len(texts)<2:
                sources[symbol]["status"]="building" if len(texts)==1 else "none"; raise ValueError("fewer than two free quarterly earnings texts")
            result=_compare(texts[0],texts[1]); signal=build_tone_signal(result,texts[0],texts[1]); signal.update({"fetched_at":utc_now(),"last_success_at":utc_now()}); cache[symbol]=clear_cache_failure(signal)
        except Exception as exc:
            cache[symbol]=cache_failure(cache.get(symbol),exc); failures[symbol]=str(exc)[:200]
        if verbose: print(f"  Free earnings tone {index}/{len(candidates)}")
    cache["_meta"]=schema_meta("stock-radar-earnings-tone-cache",schema_version=2,refreshed=len(candidates)); sources["_meta"]=schema_meta("stock-radar-earnings-text-coverage",schema_version=1)
    atomic_write_json(CACHE_PATH,cache,indent=1); atomic_write_json(SOURCE_CACHE_PATH,sources,indent=1)
    return {row["symbol"]:cache.get(row["symbol"]) for row in rows if isinstance(cache.get(row.get("symbol")),dict) and cache.get(row["symbol"],{}).get("score") is not None},{**coverage_status(rows,sources,cache),"status":"ok" if not failures else "partial","refreshed":len(candidates),"failures":failures,"model":MODEL}

def coverage_status(rows,sources,cache):
    symbols=[row.get("symbol") for row in rows if row.get("asset_type")=="company_equity"]
    statuses={"full":0,"prepared-only":0,"building":0,"none":0}
    for symbol in symbols:
        status=(sources.get(symbol) or {}).get("status","none"); statuses[status if status in statuses else "none"]+=1
    covered=statuses["full"]+statuses["prepared-only"]
    return {"eligible":len(symbols),"coverage":statuses,"coverage_pct":round(covered/max(1,len(symbols))*100,2),"signal_count":sum((cache.get(symbol) or {}).get("score") is not None for symbol in symbols)}
