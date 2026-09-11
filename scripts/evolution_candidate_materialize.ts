import { validateSandboxArtifact } from "../supabase/functions/_shared/evolution_sandbox.ts";

type FileArtifact={path:string;content:string};
type Bundle={candidate_id:string;hypothesis_id:string;parent_commit:string;branch_name:string;artifact_sha256:string;generated_by:string;generator_run_id:string;generated_at:string;files:FileArtifact[]};

function fail(message:string):never{throw new Error(`EVOLUTION_MATERIALIZER:${message}`);}
async function sha(value:string){const d=new Uint8Array(await crypto.subtle.digest("SHA-256",new TextEncoder().encode(value)));return[...d].map(b=>b.toString(16).padStart(2,"0")).join("");}
function decodeBase64Utf8(value:string){try{return new TextDecoder().decode(Uint8Array.from(atob(value),c=>c.charCodeAt(0)));}catch{fail("artifact bundle is not valid base64");}}

const encoded=(Deno.env.get("EVOLUTION_ARTIFACT_B64")??"").trim();
if(!encoded)fail("EVOLUTION_ARTIFACT_B64 missing");
let bundle:Bundle;
try{bundle=JSON.parse(decodeBase64Utf8(encoded)) as Bundle;}catch(error){fail(`bundle JSON invalid:${String(error)}`);}
if(!Array.isArray(bundle.files)||!bundle.files.length)fail("bundle files missing");
const paths=bundle.files.map(f=>String(f.path));
if(new Set(paths).size!==paths.length)fail("duplicate file paths");
const canonical=JSON.stringify(bundle.files.map(f=>({path:String(f.path),content:String(f.content)})));
const actualHash=await sha(canonical);
if(actualHash!==String(bundle.artifact_sha256).toLowerCase())fail("artifact hash mismatch");
const bytes=new TextEncoder().encode(canonical).length;
const validation=validateSandboxArtifact({
  candidateId:String(bundle.candidate_id),hypothesisId:String(bundle.hypothesis_id),parentCommit:String(bundle.parent_commit),branchName:String(bundle.branch_name),
  changedPaths:paths,patchSha256:actualHash,patchBytes:bytes,generatedBy:String(bundle.generated_by),generatorRunId:String(bundle.generator_run_id),generatedAt:String(bundle.generated_at),
});
if(!validation.valid)fail(validation.reasons.join(" | "));
if(bundle.branch_name!==Deno.env.get("EVOLUTION_BRANCH_NAME"))fail("workflow branch name does not match signed bundle");
if(bundle.parent_commit!==Deno.env.get("EVOLUTION_PARENT_SHA"))fail("workflow parent SHA does not match signed bundle");

for(const file of bundle.files){
  const path=String(file.path);const content=String(file.content);
  const slash=path.lastIndexOf("/");if(slash>0)await Deno.mkdir(path.slice(0,slash),{recursive:true});
  try{await Deno.stat(path);fail(`candidate may not overwrite existing file:${path}`);}catch(error){if(error instanceof Deno.errors.NotFound){/* expected */}else if(String(error).includes("candidate may not overwrite"))throw error;else throw error;}
  await Deno.writeTextFile(path,content);
}
console.log(JSON.stringify({status:"MATERIALIZED",candidate_id:bundle.candidate_id,branch_name:bundle.branch_name,parent_commit:bundle.parent_commit,artifact_sha256:actualHash,files:paths,bytes,canonical_mutation:false,autonomous_apply_allowed:false}));
