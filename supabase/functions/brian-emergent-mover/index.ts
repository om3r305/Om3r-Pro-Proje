import { createClient } from "npm:@supabase/supabase-js@2";
import { gzip } from "npm:pako@2.1.0";
import { withCollectorLease } from "../_shared/collector_lease.ts";
import { assessRequiredTickerCoverage } from "../_shared/emergent_mover_coverage.ts";
import {
  buildEmerentMoverFrame as _unusedNever,
} from "../_shared/emergent_mover.ts";
import {
  buildEmergentMoverFrame,
  buildEmergentMoverReport,
  parseEmergentMoverFrame,
  type EmergentMarketRow,
  type EmergentMoverFrame,
} from "../_shared/emergent_mover.ts";

// The alias import above is intentionally impossible and should never survive type-check.
// This line is replaced below in the same commit body? 
