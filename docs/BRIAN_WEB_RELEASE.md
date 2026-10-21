# BRIAN web release

Deploy from the **repository root**, using its `vercel.json`. This config copies
`monster-coins-pro/` into the static output, preserves `/api/brian/:path*` routing
to the existing Supabase functions, and includes `api/scout-feed.js`.

Deploying only the nested static folder can render the interface while omitting
the API proxy and scout function. Do not promote a build based solely on HTML/FPS.

Existing target: project `prj_3uLc3L3N9hMJgUCgzrePkAmDkE9K`, team
`team_LFZuncFjJ7gbLY3PuEDN185B`. Canonical branch: `brian-2026`.

1. Confirm commit, team/project and user release authorization; run affected tests.
2. From repository root, stage the production build:

   ```text
   vercel deploy --prod --skip-domain --yes --local-config vercel.json --project prj_3uLc3L3N9hMJgUCgzrePkAmDkE9K --scope team_LFZuncFjJ7gbLY3PuEDN185B
   ```

3. Check the staged build without credentials:

   ```text
   node tools/verify-brian-web-release.mjs https://monster-coins-pro-oemer-yildirim.vercel.app
   ```

   Protected endpoints must reject the unauthenticated request with 401; a 404
   means the deployed route/function is missing. Public heartbeat must retain
   read-only/shadow contract. Keep deployment protection enabled.

4. Verify browser interactions/data, then promote the **same** tested deployment.
5. Run the smoke check against `https://monster-coins-pro-seven.vercel.app`.
   Verify the existing authorized browser session loads Research and the other
   rooms. Do not copy dashboard credentials to a preview origin. Record actual
   source errors and waiting states; do not substitute invented data.

Neither smoke check nor opening rooms starts experiments or orders.
