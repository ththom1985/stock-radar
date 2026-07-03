"""Country of origin + flag emoji per ticker.

Derived from the Yahoo exchange suffix, with an override for foreign companies
that trade US-listed (ADRs) where the suffix would wrongly say 'USA'.
"""

# Yahoo suffix -> (country, flag)
_SUFFIX = {
    ".DE": ("Deutschland", "🇩🇪"), ".F": ("Deutschland", "🇩🇪"), ".BE": ("Belgien", "🇧🇪"),
    ".PA": ("Frankreich", "🇫🇷"), ".AS": ("Niederlande", "🇳🇱"), ".BR": ("Belgien", "🇧🇪"),
    ".L": ("UK", "🇬🇧"), ".MI": ("Italien", "🇮🇹"), ".MC": ("Spanien", "🇪🇸"),
    ".SW": ("Schweiz", "🇨🇭"), ".VX": ("Schweiz", "🇨🇭"), ".ST": ("Schweden", "🇸🇪"),
    ".OL": ("Norwegen", "🇳🇴"), ".CO": ("Dänemark", "🇩🇰"), ".HE": ("Finnland", "🇫🇮"),
    ".VI": ("Österreich", "🇦🇹"), ".WA": ("Polen", "🇵🇱"), ".IR": ("Irland", "🇮🇪"),
    ".LS": ("Portugal", "🇵🇹"), ".AT": ("Griechenland", "🇬🇷"), ".HK": ("Hongkong/China", "🇭🇰"),
    ".T": ("Japan", "🇯🇵"), ".KS": ("Südkorea", "🇰🇷"), ".KQ": ("Südkorea", "🇰🇷"),
    ".TW": ("Taiwan", "🇹🇼"), ".TWO": ("Taiwan", "🇹🇼"), ".NS": ("Indien", "🇮🇳"),
    ".BO": ("Indien", "🇮🇳"), ".SA": ("Brasilien", "🇧🇷"), ".MX": ("Mexiko", "🇲🇽"),
    ".JK": ("Indonesien", "🇮🇩"), ".KL": ("Malaysia", "🇲🇾"), ".BK": ("Thailand", "🇹🇭"),
    ".SI": ("Singapur", "🇸🇬"), ".SR": ("Saudi-Arabien", "🇸🇦"), ".JO": ("Südafrika", "🇿🇦"),
    ".AX": ("Australien", "🇦🇺"), ".NZ": ("Neuseeland", "🇳🇿"), ".TO": ("Kanada", "🇨🇦"),
    ".V": ("Kanada", "🇨🇦"), ".NE": ("Kanada", "🇨🇦"), ".CN": ("Kanada", "🇨🇦"),
}

# US-listed foreign companies (ADRs) -> (country, flag)
_ADR = {}
def _mark(flag, country, syms):
    for s in syms.split():
        _ADR[s] = (country, flag)

_mark("🇨🇳", "China", "BABA PDD JD BIDU NIO LI XPEV NTES BILI TCEHY BYDDY FUTU YMM BEKE TME ZTO "
                        "VNET GDS DQ TCOM MNSO TAL EDU VIPS")
_mark("🇹🇼", "Taiwan", "TSM UMC ASX HIMX")
_mark("🇮🇳", "Indien", "IBN HDB INFY WIT RDY MMYT")
_mark("🇧🇷", "Brasilien", "NU PBR ITUB BBD ABEV STNE XP GGB SBS BSBR VALE")
_mark("🇦🇷", "Argentinien", "MELI YPF GGAL BMA PAM")
_mark("🇲🇽", "Mexiko", "AMX FMX CX ASR PAC KOF")
_mark("🇵🇪", "Peru", "BAP")
_mark("🇰🇷", "Südkorea", "SHG CPNG")
_mark("🇸🇬", "Singapur", "SE GRAB")
_mark("🇻🇳", "Vietnam", "VFS")
_mark("🇮🇱", "Israel", "ESLT MNDY WIX GLBE")
_mark("🇿🇦", "Südafrika", "SSL GFI AU")
_mark("🇰🇿", "Kasachstan", "KSPI")
_mark("🇺🇾", "Uruguay", "DLO")
_mark("🇮🇪", "Irland", "CRH")
_mark("🇫🇮", "Finnland", "NOK")
_mark("🇸🇪", "Schweden", "ERIC")
_mark("🇨🇭", "Schweiz", "STM")
_mark("🇨🇦", "Kanada", "GOLD FNV AEM WPM KGC TECK CCJ NXE DNN BN RY TD TRP CNQ SU SHOP GIB")


def country_flag(symbol):
    """Return (country_name, flag_emoji) for a ticker."""
    s = (symbol or "").upper()
    if s in _ADR:
        return _ADR[s]
    if "." in s:
        suf = s[s.rindex("."):]
        if suf in _SUFFIX:
            return _SUFFIX[suf]
    return ("USA", "🇺🇸")
