// Behavioral tests for the actual deployed brian-fx-eye parsing/scoring logic.
// Run with: deno test --allow-read supabase/functions/brian-fx-eye

import { assertEquals } from "jsr:@std/assert@^1.0.0";
import { clip, parse, sign } from "./logic.ts";

const SAMPLE_XML = `<?xml version="1.0" encoding="UTF-8"?>
<gesmes:Envelope xmlns:gesmes="http://www.gesmes.org/xml/2002-08-01" xmlns="http://www.ecb.int/vocabulary/2002-08-01/eurofxref">
	<gesmes:subject>Reference rates</gesmes:subject>
	<gesmes:Sender>
		<gesmes:name>European Central Bank</gesmes:name>
	</gesmes:Sender>
	<Cube>
		<Cube time='2026-09-19'>
			<Cube currency='USD' rate='1.1234'/>
			<Cube currency='JPY' rate='161.23'/>
			<Cube currency='GBP' rate='0.8532'/>
			<Cube currency='CHF' rate='0.9421'/>
		</Cube>
		<Cube time='2026-09-18'>
			<Cube currency='USD' rate='1.1220'/>
			<Cube currency='JPY' rate='160.98'/>
			<Cube currency='GBP' rate='0.8540'/>
			<Cube currency='CHF' rate='0.9418'/>
		</Cube>
	</Cube>
</gesmes:Envelope>`;

Deno.test("parse: normalizes the official ECB eurofxref-hist XML shape, newest day first", () => {
  const days = parse(SAMPLE_XML);
  assertEquals(days.length, 2);
  assertEquals(days[0].date, "2026-09-19");
  assertEquals(days[0].rates, { USD: 1.1234, JPY: 161.23, GBP: 0.8532, CHF: 0.9421 });
  assertEquals(days[1].date, "2026-09-18");
  assertEquals(days[1].rates.USD, 1.1220);
});

Deno.test("parse: a Cube day with no currency children is dropped, not returned as an empty day", () => {
  const xml = `<Cube><Cube time='2026-09-19'></Cube><Cube time='2026-09-18'><Cube currency='USD' rate='1.10'/></Cube></Cube>`;
  const days = parse(xml);
  assertEquals(days.length, 1);
  assertEquals(days[0].date, "2026-09-18");
});

Deno.test("parse: no Cube day blocks at all returns an empty list, not a throw", () => {
  assertEquals(parse("<Cube></Cube>"), []);
  assertEquals(parse("not xml at all"), []);
});

Deno.test("clip: clamps to [0, 1] inclusive", () => {
  assertEquals(clip(-5), 0);
  assertEquals(clip(0), 0);
  assertEquals(clip(0.42), 0.42);
  assertEquals(clip(1), 1);
  assertEquals(clip(5), 1);
});

Deno.test("sign: returns -1/0/1, zero is neither positive nor negative", () => {
  assertEquals(sign(3.5), 1);
  assertEquals(sign(-0.001), -1);
  assertEquals(sign(0), 0);
});
