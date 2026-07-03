"""Country of origin (name + ISO-2 code for a flag image) per ticker.

Derived from the Yahoo exchange suffix, with an override for foreign companies
that trade US-listed (ADRs) where the suffix would wrongly say 'USA'.
The dashboard renders the ISO-2 code as a real flag image (flagcdn.com).
"""

# Yahoo suffix -> (country, iso2)
_SUFFIX = {
    ".DE": ("Deutschland", "de"), ".F": ("Deutschland", "de"), ".BE": ("Belgien", "be"),
    ".PA": ("Frankreich", "fr"), ".AS": ("Niederlande", "nl"), ".BR": ("Belgien", "be"),
    ".L": ("UK", "gb"), ".MI": ("Italien", "it"), ".MC": ("Spanien", "es"),
    ".SW": ("Schweiz", "ch"), ".VX": ("Schweiz", "ch"), ".ST": ("Schweden", "se"),
    ".OL": ("Norwegen", "no"), ".CO": ("Dänemark", "dk"), ".HE": ("Finnland", "fi"),
    ".VI": ("Österreich", "at"), ".WA": ("Polen", "pl"), ".IR": ("Irland", "ie"),
    ".LS": ("Portugal", "pt"), ".AT": ("Griechenland", "gr"), ".HK": ("Hongkong/China", "hk"),
    ".T": ("Japan", "jp"), ".KS": ("Südkorea", "kr"), ".KQ": ("Südkorea", "kr"),
    ".TW": ("Taiwan", "tw"), ".TWO": ("Taiwan", "tw"), ".NS": ("Indien", "in"),
    ".BO": ("Indien", "in"), ".SA": ("Brasilien", "br"), ".MX": ("Mexiko", "mx"),
    ".JK": ("Indonesien", "id"), ".KL": ("Malaysia", "my"), ".BK": ("Thailand", "th"),
    ".SI": ("Singapur", "sg"), ".SR": ("Saudi-Arabien", "sa"), ".JO": ("Südafrika", "za"),
    ".AX": ("Australien", "au"), ".NZ": ("Neuseeland", "nz"), ".TO": ("Kanada", "ca"),
    ".V": ("Kanada", "ca"), ".NE": ("Kanada", "ca"), ".CN": ("Kanada", "ca"),
}

_ADR = {}
def _mark(iso2, country, syms):
    for s in syms.split():
        _ADR[s] = (country, iso2)

_mark("cn", "China", "BABA PDD JD BIDU NIO LI XPEV NTES BILI TCEHY BYDDY FUTU YMM BEKE TME ZTO "
                      "VNET GDS DQ TCOM MNSO TAL EDU VIPS")
_mark("tw", "Taiwan", "TSM UMC ASX HIMX")
_mark("in", "Indien", "IBN HDB INFY WIT RDY MMYT")
_mark("br", "Brasilien", "NU PBR ITUB BBD ABEV STNE XP GGB SBS BSBR VALE")
_mark("ar", "Argentinien", "MELI YPF GGAL BMA PAM")
_mark("mx", "Mexiko", "AMX FMX CX ASR PAC KOF")
_mark("pe", "Peru", "BAP")
_mark("kr", "Südkorea", "SHG CPNG")
_mark("sg", "Singapur", "SE GRAB")
_mark("vn", "Vietnam", "VFS")
_mark("il", "Israel", "ESLT MNDY WIX GLBE")
_mark("za", "Südafrika", "SSL GFI AU")
_mark("kz", "Kasachstan", "KSPI")
_mark("uy", "Uruguay", "DLO")
_mark("ie", "Irland", "CRH")
_mark("fi", "Finnland", "NOK")
_mark("se", "Schweden", "ERIC")
_mark("ch", "Schweiz", "STM")
_mark("ca", "Kanada", "GOLD FNV AEM WPM KGC TECK CCJ NXE DNN BN RY TD TRP CNQ SU SHOP GIB")


def country_flag(symbol):
    """Return (country_name, iso2) for a ticker. iso2 -> flag image."""
    s = (symbol or "").upper()
    if s in _ADR:
        return _ADR[s]
    if "." in s:
        suf = s[s.rindex("."):]
        if suf in _SUFFIX:
            return _SUFFIX[suf]
    return ("USA", "us")
