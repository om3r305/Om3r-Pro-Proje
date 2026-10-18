/** A spot symbol is eligible only if the provider lists an active USDT perpetual contract. */
export function selectDerivativesUniverse(candidates: string[], exchangeInfo: unknown) {
  const payload = exchangeInfo as { symbols?: Array<Record<string, unknown>> };
  if (!Array.isArray(payload?.symbols) || !payload.symbols.length) throw new Error("USDM_CONTRACT_CATALOG_UNAVAILABLE");
  const supported = new Set(payload.symbols.filter(row => row?.status === "TRADING" &&
    row.contractType === "PERPETUAL" && row.quoteAsset === "USDT" && typeof row.symbol === "string")
    .map(row => String(row.symbol)));
  if (!supported.size) throw new Error("USDM_CONTRACT_CATALOG_INVALID");
  const unique = [...new Set(candidates)];
  return { symbols: unique.filter(symbol => supported.has(symbol)),
    skipped: unique.filter(symbol => !supported.has(symbol)) };
}
