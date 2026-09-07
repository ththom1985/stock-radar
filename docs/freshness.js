/* Dates/deadlines are derived from real bars by Python's shared market calendar. */
function sessionFreshness(data, nowMs = Date.now()) {
  const failures = [];
  const markets = new Map();
  const rows = new Map((data?.instruments || []).map(row => [row.symbol, row]));
  const contract = data?.data_status?.session_freshness;
  const seen = new Set();
  const freshSymbols = new Set();
  let fresh = 0;
  const generated = data?.generated_at;
  const awareTimestamp = value => typeof value === "string" &&
    /(?:Z|[+-]\d{2}:\d{2})$/.test(value) ? Date.parse(value) : NaN;
  const generatedMs = awareTimestamp(generated);
  if (!Number.isFinite(generatedMs)) failures.push("Ungültiger Datenzeitstempel.");
  else if (generatedMs > nowMs) failures.push("Datenzeitstempel liegt in der Zukunft.");
  if (contract?.policy !== "completed-session-v1" || contract.close_buffer_minutes !== 90 ||
      !Array.isArray(contract.groups) || !contract.groups.length || !rows.size) {
    failures.push("Kalender-Frischevertrag fehlt; neue Datenveröffentlichung erforderlich.");
    return { failures, researchFailures: failures, freshSymbols: [], freshPct: 0, markets: [] };
  }
  const minimum = Math.max(97, Number(data.data_status.coverage_sla_pct) || 97);
  for (const group of contract.groups) {
    if (!Array.isArray(group.symbols) || !group.symbols.length || !group.market) {
      failures.push("Ungültige Kalendergruppe.");
      continue;
    }
    const scope = markets.get(group.market) || {
      market: group.market, total: 0, fresh: 0, calendar: group.calendar
    };
    markets.set(group.market, scope);
    const completed = awareTimestamp(group.completed_after);
    const next = awareTimestamp(group.next_completed_after);
    const invalid = group.error || !Number.isFinite(completed) ||
      !Number.isFinite(next) || next <= completed;
    const future = !invalid && nowMs < completed;
    if (invalid) failures.push(`${group.market}: ungültiges Bar-Datum oder unbekannter Kalender.`);
    if (future) failures.push(`${group.market}: zukünftiger oder noch nicht abgeschlossener Tageskurs.`);
    for (const symbol of group.symbols) {
      if (seen.has(symbol) || rows.get(symbol)?.bar_date !== group.bar_date) {
        failures.push("Kalender-Frischevertrag passt nicht zu den Kursdaten.");
      }
      seen.add(symbol);
      scope.total += 1;
      if (!invalid && !future && nowMs < next) { scope.fresh += 1; fresh += 1; freshSymbols.add(symbol); }
    }
  }
  if (seen.size !== rows.size || [...rows.keys()].some(symbol => !seen.has(symbol))) {
    failures.push("Kalender-Frischevertrag ist unvollständig.");
  }
  const hardFailures = failures.slice();
  for (const scope of markets.values()) {
    scope.freshPct = scope.total ? scope.fresh / scope.total * 100 : 0;
    if (scope.freshPct < minimum) failures.push(
      `${scope.market}: abgeschlossene Handelssitzungen fehlen oder Bars ungültig; ` +
      `${scope.freshPct.toFixed(2)}% frisch (Minimum ${minimum.toFixed(2)}%; ${scope.calendar}).`
    );
    if (scope.freshPct < minimum) {
      for (const group of contract.groups.filter(group => group.market === scope.market)) {
        for (const symbol of group.symbols) freshSymbols.delete(symbol);
      }
    }
  }
  return { failures: [...new Set(failures)], freshPct: fresh / rows.size * 100,
    researchFailures: freshSymbols.size ? hardFailures : failures,
    freshSymbols: [...freshSymbols],
    markets: [...markets.values()] };
}

if (typeof module !== "undefined") module.exports = { sessionFreshness };
