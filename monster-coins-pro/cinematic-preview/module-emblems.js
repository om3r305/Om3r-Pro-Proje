const catalog={
 brian:['#e0e6f0','brain'],world:['#52baff','globe'],alpha:['#aa89ff','alpha'],treasury:['#ffd168','bank'],behavior:['#ff88be','pulse'],news:['#51e3ee','broadcast'],asset:['#ffa15b','asset'],
 'collector:brian-world-brain-v1':['#829dff','worldbrain'],
 'collector:brian-evolution-sandbox-v1':['#f68c91','code'],
 'collector:brian-evolution-treasury-v1':['#c0df69','vault'],
 'collector:brian-world-discovery-eye-v1':['#a0b5fc','eye'],
 'collector:brian-evolution-researcher-v1':['#df9cff','flask'],
 'collector:brian-evolution-ocean-worker-v1':['#5ad6b5','waves'],
 'collector:brian-evolution-orchestrator-v1':['#e7b693','orbit'],
 'collector:brian-alpha-decision-compiler-v2':['#f178d1','decision']
};
const paths={
 brain:'<path d="M15 8c-3-6-10-2-8 2-6 0-6 8-2 9-2 5 2 10 7 8l3-3V8m2 0c3-6 10-2 8 2 6 0 6 8 2 9 2 5-2 10-7 8l-3-3V8M10 11v5l-3 3m15-8v5l3 3M9 24l3-4m11 4-3-4"/>',
 globe:'<circle cx="16" cy="16" r="11"/><ellipse cx="16" cy="16" rx="5" ry="11"/><path d="M5 16h22M8 9h16M8 23h16"/>',
 alpha:'<path d="m5 26 11-21 11 21M10 18h12M8 26h16"/><circle cx="16" cy="11" r="1"/>',
 bank:'<path d="m4 11 12-7 12 7H4Zm3 3v10m6-10v10m6-10v10m6-10v10M4 27h24M6 24h20"/>',
 pulse:'<path d="M3 17h6l4-10 6 20 4-10h6"/><path d="M4 6h4m16 20h4"/>',
 broadcast:'<rect x="12" y="15" width="8" height="12" rx="2"/><path d="M16 15V5M9 11a10 10 0 0 0 0 10m14-10a10 10 0 0 1 0 10M5 7a15 15 0 0 0 0 18M27 7a15 15 0 0 1 0 18"/>',
 asset:'<path d="m16 3 10 13-10 13L6 16 16 3Zm-10 13h20M16 3v26"/>',
 worldbrain:'<circle cx="16" cy="16" r="11"/><path d="m7 10 5 2 1 5 5 1 3 7M19 5l-3 6 7 3 4-2M5 20l5-2"/><circle cx="18" cy="17" r="2"/>',
 code:'<path d="m11 8-8 8 8 8m10-16 8 8-8 8M19 5l-6 22"/>',
 vault:'<rect x="5" y="5" width="22" height="22" rx="3"/><circle cx="16" cy="16" r="6"/><path d="M16 10v12m-6-6h12M5 10H3m2 12H3"/>',
 eye:'<path d="M2 16s5-9 14-9 14 9 14 9-5 9-14 9S2 16 2 16Z"/><circle cx="16" cy="16" r="4"/>',
 flask:'<path d="M11 4h10m-8 0v9L5 25c-1 2 0 3 3 3h16c3 0 4-1 3-3l-8-12V4M10 20h12"/><circle cx="15" cy="23" r="1"/>',
 waves:'<path d="M3 9c4-5 9 5 13 0s9 5 13 0M3 16c4-5 9 5 13 0s9 5 13 0M3 23c4-5 9 5 13 0s9 5 13 0"/>',
 orbit:'<circle cx="16" cy="16" r="3"/><ellipse cx="16" cy="16" rx="14" ry="6" transform="rotate(35 16 16)"/><ellipse cx="16" cy="16" rx="14" ry="6" transform="rotate(-35 16 16)"/><circle cx="5" cy="7" r="2"/>',
 decision:'<path d="M7 4v9h18M16 13v10M7 13v10m18-10v10"/><rect x="3" y="23" width="8" height="5" rx="1"/><rect x="12" y="23" width="8" height="5" rx="1"/><rect x="21" y="23" width="8" height="5" rx="1"/><circle cx="7" cy="6" r="3"/>'
};
export function moduleEmblem(node){const [color,key]=catalog[node.id]||['#d9dfe8','orbit'];return {color,key,svg:`<svg viewBox="0 0 32 32" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.45" stroke-linecap="round" stroke-linejoin="round">${paths[key]}</svg>`};}
