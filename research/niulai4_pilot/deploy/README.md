# NIULAI4 pilot on a free EU cloud VM

Moves the read-only shadow pilot off the PC so it survives the PC being off. Nothing here trades,
signs or holds a wallet key. The approval (rules fingerprint `315069e3…0c37`) stays valid: the
fingerprint covers only `rules.ts` (`RULES_VERSION` + `RULES`), which this kit does not change.

## Why a VM (and not CORE / GitHub Actions)

- CORE edge functions cannot hold the pilot's long-lived websocket.
- GitHub Actions runners are in the US; Binance returns HTTP 451 there.
- The VM must be in the EU (e.g. Frankfurt) with an outbound-only firewall; no inbound port is needed
  except SSH for the operator.

## 1. Operator: create the VM (Claude cannot open accounts)

Any always-free tier works, e.g. Oracle Cloud Always Free (Frankfurt, `VM.Standard.A1.Flex`, 1 OCPU /
6 GB, Ubuntu 24.04). Requirements: 1 vCPU, 1 GB RAM, 5 GB free disk, outbound HTTPS/WSS.
Choose SSH-key login only.

## 2. Install (on the VM, as root)

    curl -fsSLO https://raw.githubusercontent.com/om3r305/Om3r-Pro-Proje/<COMMIT>/research/niulai4_pilot/deploy/setup.sh
    sudo bash setup.sh <COMMIT>

`<COMMIT>` is the pinned `brian-2026` commit to run; read the script before running it. It installs
chrony and Deno 2.1.4 (checksum-verified), checks the repo out at `<COMMIT>` into
`/opt/niulai4-pilot` (root-owned, read-only to the service), creates the `niulai4` system user and the
`niulai4-pilot` systemd unit, then runs `--selftest`. The selftest refuses until step 3 is done.

## 3. Operator: place the key and the approval (never paste the key into chat, git or a ticket)

From the PC (PowerShell), copy both files over SSH:

    scp $env:USERPROFILE\.brian\niulai4-pilot-ingest.key  <user>@<vm>:/tmp/ingest.key
    scp C:\Brian\brian-research\niulai4-pilot\APPROVED_RULES_FINGERPRINT C:\Brian\brian-research\niulai4-pilot\APPROVAL.md <user>@<vm>:/tmp/

On the VM:

    sudo install -o root -g niulai4 -m 640 /tmp/ingest.key /etc/niulai4-pilot/ingest.key && shred -u /tmp/ingest.key
    sudo install -o niulai4 -g niulai4 -m 640 /tmp/APPROVED_RULES_FINGERPRINT /tmp/APPROVAL.md /var/lib/niulai4-pilot/
    sudo -u niulai4 env DENO_DIR=/var/lib/niulai4-pilot/.deno deno run --allow-net --allow-read --allow-write --allow-run=chronyc,df /opt/niulai4-pilot/research/niulai4_pilot/main.ts --selftest --out /var/lib/niulai4-pilot

The selftest must print `ok: true` (clock skew from chrony, disk, wallet identity, fingerprint).

## 4. Switch over (one host at a time)

Two hosts must never run at once. Stop the PC pilot first (`Stop-Process -Id (Get-Content
C:\Brian\brian-research\niulai4-pilot\pilot.pid)`; the hidden Windows process gets no Ctrl+C, so its
run is closed by CORE when the next run starts), then on the VM:

    sudo systemctl enable --now niulai4-pilot
    systemctl status niulai4-pilot --no-pager
    tail -n 20 /var/lib/niulai4-pilot/pilot.stdout.log

CORE then shows a new run with `host_label = eu-vm`; the earlier PC run is closed as
`HEARTBEAT_LOST`. The PC output folder stays as the archive of the earlier runs.

## Operating

- Stop: `sudo systemctl stop niulai4-pilot` (SIGTERM → the pilot closes its run, 60 s grace).
- Crash: systemd restarts after 30 s. Exit code 2 (preflight refused) is NOT restarted; read
  `pilot.stderr.log` and fix the cause.
- Update the code: `sudo bash /opt/niulai4-pilot/research/niulai4_pilot/deploy/setup.sh <NEW_COMMIT>`
  then `sudo systemctl restart niulai4-pilot`. If `rules.ts` changed, the fingerprint changes and the
  pilot refuses until a new approval is written — that is intended.
- Remove: `sudo systemctl disable --now niulai4-pilot`, delete `/opt/niulai4-pilot`,
  `/etc/niulai4-pilot`, `/var/lib/niulai4-pilot` (after archiving its output), revoke the ingest key in
  CORE (`brian_niulai4_pilot_auth`).
