# Reproduction helpers

Run from the repository root with a Python environment containing the Strata server dependencies.
The helpers use `build-gtx1650/tuning3` as their work directory and refuse to overwrite results.
Stop any server on localhost port 18080 first. `sweep.py`, `parity.py` and `live-check.py` start
and stop their own child processes; `sanity.py` and `followup.py` use an already running server.
These are local measurement helpers, not a service manager or a production autotuner.

For the supported final profiles, prefer the launcher and benchmark commands in the report.
To use the final sweep helper in a fresh checkout:

1. Build the engine following `docs/GTX1650.md`. Use the tested prepared IQ3_XXS files.
2. Generate both `hc-q8` and `hc-q8-decode` configs with `--write-config-only --no-prompt-cache`.
3. Create `build-gtx1650/tuning3`, and copy `sanity.py` and `followup.py` from here into it.
4. Copy the built engine to `build-gtx1650/strata-residency-final` (or change the `engine` entries
   in a copy of `verified.json`). Run the server-dependency Python, not necessarily system Python:

   ```sh
   /path/to/python bench/results/2026-11-14-gtx1650ti-4gb-decode-residency/support/sweep.py \
     bench/results/2026-11-14-gtx1650ti-4gb-decode-residency/support/verified.json
   ```

`parity.py` also expects a separately built `e65fa94f` engine named
`build-gtx1650/strata-before-prefill-pool`. Do not infer old-engine parity without that build.
It forces the head to host RAM in all three arms and reads all prompts through the existing
window reader. It does not compare batched reading to window reading. `legacy-check.py`
needs the `e65fa94f` Git object but no model or GPU; it compares the old launcher via `git show`.

`live-check.py` starts the final profile with normal prompt checkpointing, runs two streaming
turns using `followup.py`, stops that server, then checks rejection of an insufficient short-read
limit. It leaves no server running.

The other JSON plans archive exploration, not a recommendation to rerun every failed arm.
`pool.json` requires the discarded `prefill-pool-prototype.patch` applied to the old source;
it does **not** test pooling on the final engine. No prototype binaries or model weights are shipped.
The reproduction helpers have checkout/Python paths generalized from the local scripts; their
syntax was checked, but the generalized copies were not used to create the archived timings.
Raw configurations use `$REPO`, `$DATA` and `$CONTROL` placeholders and are not directly runnable
until those placeholders are replaced.
