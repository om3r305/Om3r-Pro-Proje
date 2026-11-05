import { assertEquals } from "jsr:@std/assert@1";
import { classifyThemes, directlyLinkedAssets, isCompanyFiling, linkableThemes } from "./multiasset_event_link.ts";

const ev = (claim: string, source_id = "reuters.com", primary_asset: string | null = "GLOBAL_WORLD") =>
  ({ claim, source_id, primary_asset });

Deno.test("EDGAR company filings never link through words in the filer name", () => {
  // Real rows from 2026-11-05: Form 144 filings that turned gold/silver signals into 'OFFICIAL' news.
  const filing = ev("144 - BASSWOOD FINANCIAL FUND, L.P. (0001519641) (Reporting)", "official:sec:edgar_current", "GLOBAL");
  assertEquals(isCompanyFiling(filing), true);
  assertEquals(classifyThemes(filing.claim, filing.source_id), ["FINANCIAL"]); // the old path
  assertEquals(linkableThemes(filing), []);
  assertEquals(directlyLinkedAssets(filing), []);
  const goldName = ev("8-K - GOLDEN MINERALS CO (0001011509) (Filer)", "official:sec:edgar_current", "GLOBAL");
  assertEquals(directlyLinkedAssets(goldName), []);
});

Deno.test("an EDGAR filing with an explicit instrument still links directly", () => {
  const named = ev("8-K - NVIDIA CORP (0001045810) (Filer)", "official:sec:edgar_current", "equity:NVDA");
  assertEquals(directlyLinkedAssets(named), ["equity:NVDA"]);
  assertEquals(linkableThemes(named), []);
});

Deno.test("entertainment headlines link to no theme; real geopolitical and energy news still do", () => {
  assertEquals(linkableThemes(ev("God of War Laufey Delayed to May 2027 - 80 Level", "80.lv")), []);
  assertEquals(linkableThemes(ev("New war film tops the box office")), []);
  assertEquals(linkableThemes(ev("Iran says 3 vessels catch fire in Strait of Hormuz")), ["GEOPOLITICAL"]);
  assertEquals(linkableThemes(ev("Oil falls for a fourth day on lower risk of Iran war escalation")).sort(), ["ENERGY", "GEOPOLITICAL"]);
  assertEquals(linkableThemes(ev("Bank of England raises interest rate to 4%", "official:boe:news")).sort(), ["FINANCIAL", "MONETARY"]);
});

Deno.test("metals keywords and generic primary assets keep their previous behaviour outside filings", () => {
  assertEquals(directlyLinkedAssets(ev("Gold bullion reserves expanded")).sort(), ["commodity:GOLD", "etf:GLD"]);
  assertEquals(directlyLinkedAssets(ev("Silver demand rises", "reuters.com", "GLOBAL")), ["commodity:SILVER"]);
  assertEquals(directlyLinkedAssets(ev("Kraken maintenance", "official:kraken:status", "CRYPTO")), []);
});
