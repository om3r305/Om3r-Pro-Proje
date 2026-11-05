// Event -> asset relevance for the multiasset opportunity engine (operator 2026-11-05).
//
// Measured problem (7 days to 2026-11-05): 525 of 1,397 OPEN signals counted as OFFICIAL news only
// because SEC EDGAR company filings were linked to commodities through words in the filer's name
// ("144 - BASSWOOD FINANCIAL FUND" -> FINANCIAL theme -> gold/silver). Entertainment headlines leaked
// in the same way ("God of War ... delayed" -> GEOPOLITICAL -> oil). Rules here:
//   * an EDGAR company filing links only through an explicit instrument in primary_asset, never
//     through a theme or a keyword in the company name;
//   * a headline about games, films or music links to no theme;
//   * the theme vocabulary itself is unchanged.

export const EVENT_LINK_VERSION = "brian.multiasset-event-link.v1";
export const EDGAR_SOURCE_PREFIX = "official:sec:edgar";
const GENERIC_PRIMARY_ASSETS = new Set(["", "GLOBAL", "GLOBAL_WORLD", "CRYPTO"]);

export type LinkableEvent = { source_id: string; claim: string; primary_asset: string | null };

const NON_MARKET_CONTEXT =
  /\b(god of war|video ?games?|box office|movies?|films?|trailer|albums?|tv series|esports|anime)\b/i;

export function classifyThemes(claim: string, source: string): string[] {
  const x = `${claim} ${source}`.toLowerCase();
  const themes = new Set<string>();
  if (/\b(ecb|fed|federal reserve|bank of england|boe|bank of japan|boj|bank of canada|boc|interest rate|policy rate|monetary|inflation|yield|liquidity)\b/.test(x)) themes.add("MONETARY");
  if (/\b(war|missile|houthi|sanction|russia|ukraine|iran|saudi|nato|geopolit|attack|conflict|air raid)\b/.test(x)) themes.add("GEOPOLITICAL");
  if (/\b(oil|crude|brent|wti|opec|gas|pipeline|energy|refinery|blackout|power)\b/.test(x)) themes.add("ENERGY");
  if (/\b(ai|artificial intelligence|nvidia|semiconductor|chip|gemini|microsoft|meta|google|data center)\b/.test(x)) themes.add("AI_TECH");
  if (/\b(bank|banking|credit|financial|systemic|treasury|bond|debt|liquidity)\b/.test(x)) themes.add("FINANCIAL");
  if (/\b(dollar|usd|euro|eur|yen|jpy|sterling|gbp|currency|fx)\b/.test(x)) themes.add("FX");
  return [...themes];
}

export function isCompanyFiling(event: LinkableEvent): boolean {
  return String(event.source_id ?? "").startsWith(EDGAR_SOURCE_PREFIX);
}

function explicitInstrument(event: LinkableEvent): string | null {
  const asset = String(event.primary_asset ?? "").trim();
  return GENERIC_PRIMARY_ASSETS.has(asset) || !asset.includes(":") ? null : asset;
}

/** Instruments the event names directly (primary asset, or an explicit metals reference). */
export function directlyLinkedAssets(event: LinkableEvent): string[] {
  const assets = new Set<string>();
  const explicit = explicitInstrument(event);
  if (explicit) assets.add(explicit);
  if (isCompanyFiling(event)) return [...assets];
  if (event.primary_asset && !GENERIC_PRIMARY_ASSETS.has(event.primary_asset)) assets.add(event.primary_asset);
  // Explicit metals references also work when the headline has no macro theme.
  if (/\b(gold|bullion)\b/i.test(event.claim)) { assets.add("commodity:GOLD"); assets.add("etf:GLD"); }
  if (/\bsilver\b/i.test(event.claim)) assets.add("commodity:SILVER");
  return [...assets];
}

/** Themes through which the event may reach every asset tagged with that theme. */
export function linkableThemes(event: LinkableEvent): string[] {
  if (isCompanyFiling(event)) return [];
  if (NON_MARKET_CONTEXT.test(String(event.claim ?? ""))) return [];
  return classifyThemes(String(event.claim ?? ""), String(event.source_id ?? ""));
}
