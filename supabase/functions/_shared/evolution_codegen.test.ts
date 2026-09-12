import { canGenerateBuiltIn, generateBuiltInArtifact } from "./evolution_codegen.ts";
import type { HypothesisCandidate } from "./evolution_research.ts";

function hypothesis(kind:HypothesisCandidate["hypothesisKind"]):HypothesisCandidate{return{
  hypothesisId:`h-${kind}`,observedAt:"2026-09-11T13:00:00Z",problemStatement:"measured weakness",proposedMechanism:"bounded challenger",
  targetCapabilities:["alpha.expected-edge"],evidenceRefs:["e:1"],counterEvidenceRefs:[],measurableSuccessCriteria:["prospective improvement"],
  stage:"RESEARCHING",uncertainty:.3,priority:.9,hypothesisKind:kind,metadata:{},
};}
const parent="ba330f4ec4bd3d76b1fe5564d6b4e95cd41ef4f0";

Deno.test("built-in generator covers bounded research kinds",()=>{for(const kind of ["ACTION_GATE","EXPECTED_EDGE","RELIABILITY_FEEDBACK","COST_CONTROL","DRIFT"] as const){if(!canGenerateBuiltIn(hypothesis(kind)))throw new Error(`missing generator ${kind}`);const a=generateBuiltInArtifact(hypothesis(kind),parent);if(a.files.length!==3)throw new Error(`bad file count ${kind}`);if(a.files.some(f=>!a.brief.changedPaths.includes(f.path)))throw new Error(`path mismatch ${kind}`);if(a.brief.autonomousApplyAllowed||!a.brief.requiredHumanReview)throw new Error("unsafe artifact brief");}});
Deno.test("built-in generated code stays in isolated candidate namespace",()=>{const a=generateBuiltInArtifact(hypothesis("EXPECTED_EDGE"),parent);for(const f of a.files){if(!(f.path.startsWith("supabase/functions/_shared/evolution_candidates/")||f.path.startsWith("docs/evolution_candidates/")))throw new Error(f.path);}if(!a.files[0].content.includes("SHADOW ONLY"))throw new Error("shadow declaration missing");});
Deno.test("capability gaps require a specialized or external generator",()=>{const h=hypothesis("CAPABILITY_GAP");if(canGenerateBuiltIn(h))throw new Error("generic capability gap should not get arbitrary code template");let threw=false;try{generateBuiltInArtifact(h,parent);}catch{threw=true;}if(!threw)throw new Error("unsupported generator did not fail closed");});
