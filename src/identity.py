"""Conservative identity, listing and jurisdiction context.

Yahoo's ``country`` field is treated as provider headquarters/location metadata,
never as proof of legal incorporation. Legal domicile is populated only from the
small, cited override table below. Economic exposure remains a separate heuristic.
"""
from __future__ import annotations

from typing import Any


MODEL_STATUS = "heuristic_unvalidated"

_CHINA_EXPOSURE = {
    "BABA", "PDD", "JD", "BIDU", "NIO", "LI", "XPEV", "NTES", "BILI",
    "TCEHY", "BYDDY", "FUTU", "YMM", "BEKE", "TME", "ZTO", "VNET", "GDS",
    "DQ", "TCOM", "MNSO", "TAL", "EDU", "VIPS", "STG",
}
_BRAZIL_EXPOSURE = {
    "PBR", "ITUB", "BBD", "ABEV", "STNE", "XP", "GGB", "SBS", "BSBR", "VALE",
}
_ARGENTINA_EXPOSURE = {"YPF", "GGAL", "BMA", "PAM"}
_BRAZIL_STATE_LINKED = {"PBR", "SBS"}

# Verified from the named 2026 SEC Form 20-F filing. No other legal domicile is
# inferred from headquarters, listing venue, name, or economic exposure.
_VERIFIED_LEGAL_DOMICILE = {
    "BABA": {
        "country": "Cayman Islands",
        "code": "KY",
        "source": "SEC Form 20-F accession 0001193125-26-231755, filed 2026-05-20",
        "structure_context": "known_china_adr_vie",
    },
    "PDD": {
        "country": "Cayman Islands",
        "code": "KY",
        "source": "SEC Form 20-F accession 0001104659-26-050727, filed 2026-04-29",
        "structure_context": "known_china_adr_vie",
    },
    "TCOM": {
        "country": "Cayman Islands",
        "code": "KY",
        "source": "SEC Form 20-F accession 0001193125-26-183379, filed 2026-04-28",
        "structure_context": "known_china_adr_vie",
    },
}

_COUNTRY_ALIASES: dict[str, tuple[str, str | None, str | None]] = {}


def _register_country(
    stable: str,
    code: str,
    region: str,
    *aliases: str,
) -> None:
    for value in (stable, *aliases):
        _COUNTRY_ALIASES[value.casefold()] = (stable, code, region)


_register_country("Argentinien", "AR", "Lateinamerika", "Argentina")
_register_country("Australien", "AU", "Asien-Pazifik", "Australia")
_register_country("Belgien", "BE", "Europa", "Belgium")
_register_country("Bermuda", "BM", "Karibik")
_register_country("Brasilien", "BR", "Lateinamerika", "Brazil")
_register_country("Cayman Islands", "KY", "Karibik")
_register_country("Chile", "CL", "Lateinamerika")
_register_country("China", "CN", "China")
_register_country("Dänemark", "DK", "Europa", "Denmark")
_register_country("Deutschland", "DE", "Europa", "Germany")
_register_country("Finnland", "FI", "Europa", "Finland")
_register_country("Frankreich", "FR", "Europa", "France")
_register_country("Griechenland", "GR", "Europa", "Greece")
_register_country("Hongkong", "HK", "China/Hongkong", "Hong Kong", "Hongkong/China")
_register_country("Indien", "IN", "Asien", "India")
_register_country("Indonesien", "ID", "Asien", "Indonesia")
_register_country("Irland", "IE", "Europa", "Ireland")
_register_country("Israel", "IL", "Nahost")
_register_country("Italien", "IT", "Europa", "Italy")
_register_country("Japan", "JP", "Asien")
_register_country("Kanada", "CA", "Nordamerika", "Canada")
_register_country("Kasachstan", "KZ", "Zentralasien", "Kazakhstan")
_register_country("Luxemburg", "LU", "Europa", "Luxembourg")
_register_country("Malaysia", "MY", "Asien")
_register_country("Mexiko", "MX", "Lateinamerika", "Mexico")
_register_country("Neuseeland", "NZ", "Asien-Pazifik", "New Zealand")
_register_country("Niederlande", "NL", "Europa", "Netherlands")
_register_country("Norwegen", "NO", "Europa", "Norway")
_register_country("Österreich", "AT", "Europa", "Austria")
_register_country("Peru", "PE", "Lateinamerika")
_register_country("Polen", "PL", "Europa", "Poland")
_register_country("Portugal", "PT", "Europa")
_register_country("Saudi-Arabien", "SA", "Nahost", "Saudi Arabia")
_register_country("Schweden", "SE", "Europa", "Sweden")
_register_country("Schweiz", "CH", "Europa", "Switzerland")
_register_country("Singapur", "SG", "Asien", "Singapore")
_register_country("Spanien", "ES", "Europa", "Spain")
_register_country("Südafrika", "ZA", "Afrika", "South Africa")
_register_country("Südkorea", "KR", "Asien", "South Korea")
_register_country("Taiwan", "TW", "Asien")
_register_country("Thailand", "TH", "Asien")
_register_country("Uruguay", "UY", "Lateinamerika")
_register_country("USA", "US", "Nordamerika", "United States")
_register_country(
    "Vereinigtes Königreich",
    "GB",
    "Europa",
    "United Kingdom",
    "UK",
)
_register_country("Vietnam", "VN", "Asien")

# Every suffix covered by src.markets._PROFILES has an explicit listing mapping.
_LISTING_SUFFIX = {
    ".DE": ("Frankfurt/Xetra (.DE)", "Deutschland", "DE"),
    ".F": ("Frankfurt (.F)", "Deutschland", "DE"),
    ".PA": ("Euronext Paris (.PA)", "Frankreich", "FR"),
    ".AS": ("Euronext Amsterdam (.AS)", "Niederlande", "NL"),
    ".BR": ("Euronext Brüssel (.BR)", "Belgien", "BE"),
    ".MI": ("Borsa Italiana (.MI)", "Italien", "IT"),
    ".MC": ("Madrid (.MC)", "Spanien", "ES"),
    ".VI": ("Wien (.VI)", "Österreich", "AT"),
    ".HE": ("Helsinki (.HE)", "Finnland", "FI"),
    ".LS": ("Lissabon (.LS)", "Portugal", "PT"),
    ".IR": ("Dublin (.IR)", "Irland", "IE"),
    ".L": ("London (.L)", "Vereinigtes Königreich", "GB"),
    ".SW": ("SIX Swiss Exchange (.SW)", "Schweiz", "CH"),
    ".ST": ("Stockholm (.ST)", "Schweden", "SE"),
    ".OL": ("Oslo (.OL)", "Norwegen", "NO"),
    ".CO": ("Kopenhagen (.CO)", "Dänemark", "DK"),
    ".WA": ("Warschau (.WA)", "Polen", "PL"),
    ".AT": ("Athen (.AT)", "Griechenland", "GR"),
    ".T": ("Tokio (.T)", "Japan", "JP"),
    ".HK": ("Hongkong (.HK)", "Hongkong", "HK"),
    ".KS": ("Korea Exchange (.KS)", "Südkorea", "KR"),
    ".KQ": ("KOSDAQ (.KQ)", "Südkorea", "KR"),
    ".TW": ("Taiwan Stock Exchange (.TW)", "Taiwan", "TW"),
    ".TWO": ("Taipei Exchange (.TWO)", "Taiwan", "TW"),
    ".NS": ("NSE Indien (.NS)", "Indien", "IN"),
    ".BO": ("BSE Indien (.BO)", "Indien", "IN"),
    ".JK": ("Jakarta (.JK)", "Indonesien", "ID"),
    ".KL": ("Kuala Lumpur (.KL)", "Malaysia", "MY"),
    ".BK": ("Bangkok (.BK)", "Thailand", "TH"),
    ".SI": ("Singapur (.SI)", "Singapur", "SG"),
    ".SR": ("Riad (.SR)", "Saudi-Arabien", "SA"),
    ".JO": ("Johannesburg (.JO)", "Südafrika", "ZA"),
    ".SA": ("São Paulo (.SA)", "Brasilien", "BR"),
    ".MX": ("Mexiko (.MX)", "Mexiko", "MX"),
    ".AX": ("Australian Securities Exchange (.AX)", "Australien", "AU"),
    ".NZ": ("New Zealand Exchange (.NZ)", "Neuseeland", "NZ"),
    ".TO": ("Toronto (.TO)", "Kanada", "CA"),
    ".V": ("TSX Venture (.V)", "Kanada", "CA"),
    ".NE": ("Cboe Canada (.NE)", "Kanada", "CA"),
    ".CN": ("Canadian Securities Exchange (.CN)", "Kanada", "CA"),
}

_VENUE_COUNTRY = {
    "us": ("US-Markt", "USA", "US"),
    "nasdaq": ("NASDAQ", "USA", "US"),
    "nyse": ("NYSE", "USA", "US"),
    "nyse american": ("NYSE American", "USA", "US"),
    "us-adr": ("US-ADR", "USA", "US"),
    "otc": ("US OTC", "USA", "US"),
    "xetra": ("Xetra", "Deutschland", "DE"),
    "nse": ("NSE Indien", "Indien", "IN"),
    "london": ("London", "Vereinigtes Königreich", "GB"),
    "euronext paris": ("Euronext Paris", "Frankreich", "FR"),
    "paris": ("Paris", "Frankreich", "FR"),
    ".pa=paris": ("Euronext Paris", "Frankreich", "FR"),
    "tokio": ("Tokio", "Japan", "JP"),
    ".t=tokio": ("Tokio", "Japan", "JP"),
    "hongkong": ("Hongkong", "Hongkong", "HK"),
    "euronext amsterdam": ("Euronext Amsterdam", "Niederlande", "NL"),
    "wien": ("Wien", "Österreich", "AT"),
    "mailand": ("Mailand", "Italien", "IT"),
    "oslo": ("Oslo", "Norwegen", "NO"),
    "six": ("SIX Swiss Exchange", "Schweiz", "CH"),
    "jakarta": ("Jakarta", "Indonesien", "ID"),
    "tsx venture": ("TSX Venture", "Kanada", "CA"),
    "madrid": ("Madrid", "Spanien", "ES"),
    ".mc=madrid": ("Madrid", "Spanien", "ES"),
    "kopenhagen": ("Kopenhagen", "Dänemark", "DK"),
    "stockholm": ("Stockholm", "Schweden", "SE"),
    "australien": ("Australian Securities Exchange", "Australien", "AU"),
    ".ax=australien": ("Australian Securities Exchange", "Australien", "AU"),
    "warschau": ("Warschau", "Polen", "PL"),
    "toronto": ("Toronto", "Kanada", "CA"),
    "helsinki": ("Helsinki", "Finnland", "FI"),
    ".he=helsinki": ("Helsinki", "Finnland", "FI"),
    "korea": ("Korea Exchange", "Südkorea", "KR"),
    ".ks=korea": ("Korea Exchange", "Südkorea", "KR"),
    "johannesburg": ("Johannesburg", "Südafrika", "ZA"),
    "athen": ("Athen", "Griechenland", "GR"),
    "riad": ("Riad", "Saudi-Arabien", "SA"),
}

_EMERGING_CODES = {"AR", "BR", "CN", "IN", "ID", "KZ", "MX", "SA", "ZA"}


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = " ".join(value.split()).strip()
    if not value or "\ufffd" in value or value.casefold() in {"n/a", "none", "unknown"}:
        return None
    return value


def normalize_country(value: Any) -> tuple[str | None, str | None, str | None, str]:
    """Return stable name/code/region and classification status."""
    cleaned = _clean_text(value)
    if not cleaned:
        return None, None, None, "unavailable"
    known = _COUNTRY_ALIASES.get(cleaned.casefold())
    if known:
        return *known, "normalized"
    return cleaned, None, "Nicht klassifiziert", "unclassified"


def _listing_context(
    row: dict[str, Any],
) -> tuple[str | None, str | None, str | None, str]:
    symbol = str(row.get("symbol") or "").upper()
    configured = _clean_text(row.get("listing_market") or row.get("exchange"))
    for suffix, (market, country, code) in sorted(
        _LISTING_SUFFIX.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if symbol.endswith(suffix):
            return market, country, code, "ticker_suffix"
    if symbol.endswith("-USD"):
        return configured or "Krypto-Referenzmarkt", None, None, "instrument_type"
    if "." in symbol:
        suffix = symbol[symbol.rindex(".") :]
        return configured or suffix, None, None, f"unknown_suffix:{suffix}"
    if configured:
        known = _VENUE_COUNTRY.get(configured.casefold())
        if known:
            market, country, code = known
            return market, country, code, "configured_market"
        return configured, None, None, "unclassified_configured_market"
    return None, None, None, "unavailable"


def _economic_exposure(
    row: dict[str, Any],
) -> tuple[str, str | None, str, str, str]:
    """Use only explicit exposure evidence; headquarters is never a proxy."""
    symbol = str(row.get("symbol") or "").upper()
    if symbol in _CHINA_EXPOSURE:
        return "China", "CN", "China", "documented_ticker_override", "classified"
    if symbol in _BRAZIL_EXPOSURE:
        return "Brasilien", "BR", "Lateinamerika", "documented_ticker_override", "classified"
    if symbol in _ARGENTINA_EXPOSURE:
        return "Argentinien", "AR", "Lateinamerika", "documented_ticker_override", "classified"
    if symbol.endswith(".HK"):
        # Explicit documented rule: a Hong Kong equity listing is treated as
        # China/Hong Kong exposure context, not merely as a headquarters proxy.
        return "China", "CN", "China/Hongkong", "hong_kong_listing_override", "classified"
    return "Nicht verfügbar", None, "Nicht verfügbar", "unavailable", "unavailable"


def _group(inputs: list[str], missing: list[str] | None = None) -> dict[str, Any]:
    return {
        "model_status": MODEL_STATUS,
        "actionable": False,
        "inputs_used": inputs,
        "missing_inputs": list(missing or []),
    }


def _jurisdiction_risk(
    row: dict[str, Any],
    headquarters_country: str | None,
    legal: dict[str, Any] | None,
    listing_country: str | None,
    listing_code: str | None,
    exposure_country: str,
    exposure_code: str | None,
    exposure_region: str,
    exposure_status: str,
) -> dict[str, Any]:
    symbol = str(row.get("symbol") or "").upper()
    reasons: list[str] = []
    penalty = 0
    level = "low"

    if exposure_code == "CN":
        us_listed = listing_code == "US"
        penalty = 18 if us_listed else 14
        level = "high"
        reasons.extend(
            [
                "China-Exposure: regulatorische und politische Eingriffe sowie Datenregulierung können Geschäftsmodell und Kapitalzugang verändern.",
                "Kapitalverkehrskontrollen und staatlicher Einfluss begrenzen die Vergleichbarkeit mit Märkten ohne diese Struktur.",
                "Geopolitik, Sanktionen sowie Prüfungs-/Transparenzkonflikte können einen strukturellen Risikoaufschlag und niedrigere Bewertungsmultiplikatoren erklären.",
            ]
        )
        if legal and legal.get("structure_context") == "known_china_adr_vie" and us_listed:
            reasons.append(
                f"Verifizierter juristischer Sitz Cayman Islands ({legal['source']}); bekannter US-ADR/VIE-Kontext mit gesondert zu prüfender Eigentums- und Delistingstruktur."
            )
        elif str(row.get("listing_market") or "").casefold() == "us-adr":
            reasons.append(
                "Konfigurierter US-ADR-Kontext: Verwahr-, Rechts- und Delistingstruktur instrumentspezifisch prüfen; kein Cayman-/VIE-Schluss ohne separate Evidenz."
            )
        elif us_listed:
            reasons.append(
                "US-gehandeltes China-Exposure: Prüfungs-, Transparenz- und Delistingrisiken bleiben; keine VIE- oder Cayman-Aussage ohne verifizierte Evidenz."
            )
        else:
            reasons.append(
                "Hongkong/China-Notierung: kein US-ADR-Abzug; lokale Markt-, Kapitalfluss- und Rechtsrisiken bleiben."
            )
    elif exposure_code == "AR":
        penalty = 14
        level = "high"
        reasons.extend(
            [
                "Argentinien-Exposure: Kapitalverkehrskontrollen und Währungsumrechnung können Ausschüttungen und Vergleichswerte verzerren.",
                "Hohe Inflation sowie abrupte Wirtschafts- und Regulierungspolitik erhöhen Ergebnis- und Bewertungsrisiken.",
            ]
        )
    elif exposure_code == "BR":
        penalty = 11 if symbol in _BRAZIL_STATE_LINKED else 8
        level = "high" if symbol in _BRAZIL_STATE_LINKED else "medium"
        reasons.append(
            "Brasilien-Exposure: BRL-Währungsschwankungen und Governance-/Politikrisiken können den Bewertungsabschlag mit erklären."
        )
        if symbol in _BRAZIL_STATE_LINKED:
            reasons.append(
                "Dokumentierter staatlicher Einfluss: Preis-, Investitions- oder Ausschüttungspolitik kann Minderheitsaktionärsinteressen überlagern."
            )
    elif exposure_code in _EMERGING_CODES:
        penalty = 5
        level = "medium"
        reasons.append(
            f"Explizites {exposure_country}-Exposure: Wechselkurs-, Governance- und Kapitalmarktrisiken werden als begrenzter Kontextabschlag erfasst."
        )
    elif exposure_status != "classified":
        level = "unknown"
        reasons.append(
            f"Wirtschaftliches Exposure {exposure_country!r} ist nicht klassifiziert; kein Risikoabschlag mangels belastbarer Zuordnung."
        )
    else:
        reasons.append(
            "Aus den vorhandenen Exposure-Metadaten ergibt sich kein spezifischer erhöhter Jurisdiktionsabschlag."
        )

    return {
        **_group(
            [
                "documented ticker exposure overrides",
                "provider headquarters country",
                "verified legal-domicile overrides",
                "listing market",
                "ticker suffix",
            ],
            [] if exposure_status == "classified" else ["classified economic exposure unavailable"],
        ),
        "level": level,
        "score": penalty,
        "penalty_points": penalty,
        "bounded_range": [0, 20],
        "reasons": reasons,
        "economic_exposure": {"country": exposure_country, "region": exposure_region},
        "headquarters_country": headquarters_country,
        "legal_domicile": legal.get("country") if legal else None,
        "legal_domicile_source": legal.get("source") if legal else None,
        "listing_country": listing_country,
        "heuristic_note": (
            "Transparenter, begrenzter Kontextabschlag; keine mathematisch bewiesene "
            "Bewertung und kein DCF."
        ),
    }


def enrich_identity(row: dict[str, Any]) -> dict[str, Any]:
    """Attach identity fields without conflating headquarters and domicile."""
    symbol = str(row.get("symbol") or "").upper()
    short_name = _clean_text(row.get("short_name") or row.get("name"))
    provider_name = _clean_text(row.get("provider_long_name"))
    display_name = provider_name or short_name or symbol
    display_source = (
        "provider_long_name" if provider_name else "short_name" if short_name else "symbol"
    )

    provider_country_raw = _clean_text(
        row.get("provider_country_raw") or row.get("provider_country")
    )
    (
        headquarters_country,
        headquarters_code,
        _headquarters_region,
        headquarters_status,
    ) = normalize_country(provider_country_raw)
    legal = _VERIFIED_LEGAL_DOMICILE.get(symbol)
    listing_market, listing_country, listing_code, listing_source = _listing_context(row)
    (
        exposure_country,
        exposure_code,
        exposure_region,
        exposure_source,
        exposure_status,
    ) = _economic_exposure(
        row,
    )
    sector = _clean_text(row.get("sector"))
    industry = _clean_text(row.get("industry"))

    complete = bool(
        provider_name
        and headquarters_country
        and listing_country
        and sector
        and industry
        and exposure_status == "classified"
    )
    partial = bool(display_name and (headquarters_country or listing_country))
    legal_country = legal.get("country") if legal else None
    legal_code = legal.get("code") if legal else None
    row.update(
        {
            "short_name": short_name,
            "display_name_full": display_name,
            "provider_country_raw": provider_country_raw,
            "provider_country": headquarters_country,
            "provider_country_code": headquarters_code,
            "provider_country_normalization_status": headquarters_status,
            "headquarters_country": headquarters_country,
            "headquarters_country_code": headquarters_code,
            "legal_domicile": legal_country,
            "legal_domicile_code": legal_code,
            "legal_domicile_verified": legal is not None,
            "legal_domicile_source": legal.get("source") if legal else None,
            # Backward-compatible alias: now only verified domicile, never HQ.
            "issuer_country": legal_country,
            "issuer_country_code": legal_code,
            "listing_market": listing_market,
            "listing_country": listing_country,
            "listing_country_code": listing_code,
            "economic_exposure_country": exposure_country,
            "economic_exposure_country_code": exposure_code,
            "economic_exposure_region": exposure_region,
            "economic_exposure_source": exposure_source,
            "economic_exposure_classification_status": exposure_status,
            "jurisdiction_code": exposure_code,
            "sector_display": sector or "Nicht verfügbar",
            "industry_display": industry or "Nicht verfügbar",
            "identity_source": {
                "display_name": display_source,
                "headquarters_country": "provider_country" if provider_country_raw else "unavailable",
                "legal_domicile": "verified_override" if legal else "unavailable",
                "listing": listing_source,
                "economic_exposure": exposure_source,
            },
            "identity_semantics": {
                "provider_country": "normalized provider headquarters/location; not legal domicile",
                "headquarters_country": "explicit semantic alias of normalized provider_country",
                "legal_domicile": "nullable; populated only from cited verified filing override",
                "issuer_country": "DEPRECATED alias of legal_domicile; never provider headquarters",
                "listing_country": "exchange country; not issuer domicile or economic exposure",
            },
            "identity_status": "complete" if complete else "partial" if partial else "fallback",
        }
    )
    row["jurisdiction_risk"] = _jurisdiction_risk(
        row,
        headquarters_country,
        legal,
        listing_country,
        listing_code,
        exposure_country,
        exposure_code,
        exposure_region,
        exposure_status,
    )
    return row
