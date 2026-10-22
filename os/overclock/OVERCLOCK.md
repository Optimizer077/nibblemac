# Overclock test (GitHub issue #10)

Does a 2.0 GHz ARM clock speed pinfer up on this Pi 4, and is it stable? Decode is DRAM-bound
(424.9 MB of weights per token at 4.20 GB/s), so the clock should mostly help the compute-bound
parts: prefill (prompt reading), the speculative verify pass and Cerno 4B decisions, which are
prefill only.

**Board:** Raspberry Pi 4 Model B Rev 1.1 (revision `c03111`, BCM2711 B0), 4 GB, fan-cooled. Stock
1500 MHz; `arm_boost=1` in `config.txt` has no effect on this revision (measured 1500 MHz).

**Approved by the owner (2026-10-22):** `arm_freq=2000`, `over_voltage=6`, no `force_turbo`. Per
the Raspberry Pi `config.txt` documentation, the warranty bit is set only by `force_turbo=1` together
with `over_voltage` above 0, and the firmware caps the clock at the 85 °C `temp_limit`. The scripts
refuse values above the approved ones.

## Procedure
The test boot uses the firmware's tryboot mechanism: `config.txt` is never edited, and the next reboot or
power cycle is a normal 1.5 GHz boot. `tryboot.txt` normally holds the custom-kernel test boot
(`docs/pi-state.md` change #4); `oc_tryboot.sh` backs it up and puts it back.

```bash
# 0. stock clock, same binary: the baseline (each bench holds the Pi-wide lock)
python harness/pi.py push os/overclock/oc_tryboot.sh os/overclock/oc_soak.sh os/overclock/oc_measure.sh --dest os/overclock
python harness/pi.py run "mkdir -p data/cerno && cp tmp/cerno/ids/p64.ids tmp/cerno/ids/p256.ids data/cerno/"
python harness/pi.py bench oc-1500-short -- bash os/overclock/oc_measure.sh ./engine/pinfer_run_fast 3 short
python harness/pi.py bench oc-1500-ctx   -- bash os/overclock/oc_measure.sh ./engine/pinfer_run_fast 1 ctx
python harness/pi.py bench oc-1500-cerno -- bash os/overclock/oc_measure.sh ./engine/pinfer_run_fast 3 cerno
# 1. snapshot, arm, one-time boot (all other agents off the Pi: the reboot drops the lock)
python harness/pi.py snapshot before-overclock
python harness/pi.py run "sudo bash os/overclock/oc_tryboot.sh arm 2000 6"
python harness/pi.py run "sudo reboot '0 tryboot'"
python harness/pi.py run "bash os/overclock/oc_tryboot.sh status"     # tryboot 1, arm_freq 2000
# 2. stability: 5 min of 4-core prefill load, then the exactness gate
python harness/pi.py bench oc-soak-2000 -- bash os/overclock/oc_soak.sh 300 ./engine/pinfer_run_fast \
    -m models/qwen35-0.8b-q4k-headq4k.pq -t 4 --ids-file data/ctx2048.ids -n 1 --bench
python harness/pi.py bench oc-spec-exact-2000 -- bash engine/spec_exact.sh ./engine/pinfer_run_fast
# 3. the same measurements as step 0
python harness/pi.py bench oc-2000-short -- bash os/overclock/oc_measure.sh ./engine/pinfer_run_fast 3 short
python harness/pi.py bench oc-2000-ctx   -- bash os/overclock/oc_measure.sh ./engine/pinfer_run_fast 1 ctx
python harness/pi.py bench oc-2000-cerno -- bash os/overclock/oc_measure.sh ./engine/pinfer_run_fast 3 cerno
# 4. back to normal
python harness/pi.py run "sudo bash os/overclock/oc_tryboot.sh restore"
python harness/pi.py run "sudo reboot"
python harness/pi.py run "bash os/overclock/oc_tryboot.sh status"     # tryboot 0, arm_freq 1500
```

**Pass criteria:**
- **Soak:** `runs_failed` 0, `throttled_or` `0x0` (no under-voltage, frequency cap or thermal limit, not
  even briefly), the load at 2000 MHz, `max_temp_c` below 80 °C.
- **Exactness:** `spec_exact` all `"match":true`, the original 20 rows with their baseline md5s. An
  unstable clock shows up as a crash or a changed output.
- **Measurements:** only runs whose label line shows `"throttled":"0x0"` and `"ok":true` count, and only
  from `valid=true` result files.

**If it fails:** a kernel panic reboots by itself after 10 s (`panic=10` in `cmdline-oc.txt`) into the normal
configuration. A hard freeze needs a power cycle, which also boots normally. Then run step 4's
`restore` to put the custom-kernel `tryboot.txt` back.

**Undo:** `sudo bash os/overclock/oc_tryboot.sh restore && sudo reboot`. This restores `tryboot.txt` from
`tryboot.txt.pre-oc` and deletes `cmdline-oc.txt`; `config.txt`, `cmdline.txt` and `kernel8.img` are never
written (the script checks their SHA-256 before and after).

## Results

### Run 1, 2026-10-22: 2000 MHz / `over_voltage=6` — FAILED (hard freeze under load)
| time (KST) | step | result | evidence |
|---|---|---|---|
| 17:39 | pause window opened; 9 earlier locked jobs drained first | — | `harness/pi.pause` |
| 17:54–18:02 | baselines at 1500 MHz (same binaries) | Cerno 4B prefill p64 / p256 6.83 / 6.84 tok/s; 5 Cerno requests (13 questions) decision mean 20.79 s, median 19.35 s, first question 12.33 s, later 5.54 s, 6.87 tok/s, 2306.6 MiB; 0.8B plain 7.83, default speculative 20.35, chat27 10.91, ctx512 prefill 41.6 tok/s; throttled 0x0 throughout | `bench/results/oc-1500-cerno_20261022_175728.json`, `oc-1500-decide_20261022_175921.json`, `oc-1500-short_20261022_180155.json` |
| 18:02 | snapshot, `oc_tryboot.sh arm 2000 6`, `reboot '0 tryboot'` | back in ~10 s; tryboot 1, `arm_freq=2000`, `over_voltage=6`, `force_turbo=0`, 2000.5 MHz, 1.0187 V, 48.7 °C, throttled 0x0, `panic=10` on the cmdline | `logs/snapshots/20261022_180207_before-overclock`, `logs/runs/20261022_1803*` |
| 18:03:45 | 5-min soak (`oc_soak.sh 300`, 0.8B 2048-token prefill loop, 4 threads) | **ssh dropped after ~27 s**; no answer on the cable or Wi-Fi for 6+ min, so no panic reboot (`panic=10` would have rebooted within ~1 min): a hard freeze | `logs/harness.jsonl` 18:03:45 (bench exit 1, ssh connect timeout) |

Verdict: **2.0 GHz with `over_voltage=6` is not stable on this board** (Pi 4 Model B Rev 1.1, BCM2711 B0).
Nothing was measured at 2.0 GHz.
A lower setting (for example 1.8 GHz with a smaller `over_voltage`) would be a new configuration and needs
the owner's OK first.
