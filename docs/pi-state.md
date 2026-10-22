# Pi state: every change from stock, and how to undo it

The Pi must always be reproducible from: the stock image + this list. Before a change, take
`python harness/pi.py snapshot before-<change>`; after it, add a row here and a CHANGELOG entry.

## Device
| Item | Value |
|---|---|
| Board | Raspberry Pi 4 Model B Rev 1.1, 4 GB |
| Boot disk | Crucial X6 1 TB USB SSD (PC disk 2 when attached), `/dev/sda` on the Pi |
| Base image | `2026-10-06-raspios-trixie-arm64-lite.img.xz`, SHA-256 `483db18a48da399b5b7022ffae9a07bc6d99daab30e593aedac09b69ff422843` |
| Flashed | 2026-10-21 with `os/stock/flash_disk.ps1` |
| Network | **Control:** direct PC↔Pi Ethernet cable, Pi `10.42.0.2/24`, PC `10.42.0.1/24`, no gateway. **Internet:** Wi-Fi <wifi-ssid>, 192.168.1.166 (DHCP). Hostname `pi4` (`pi4.local` does **not** resolve correctly from the PC) |
| Login | `pi`, SSH key `~/.ssh/id_ed25519_desktop` only; console password in `harness/pi.secret` |

## Changes from stock
| # | Date | Change | Where on the Pi | Undo | Evidence |
|---|---|---|---|---|---|
| 0 | 2026-10-21 | First-boot config via cloud-init: hostname pi4, user pi (key-only SSH, passwordless sudo), Wi-Fi PSK hash, Ethernet DHCP, timezone Asia/Seoul, `cfg80211.ieee80211_regdom=KR` in cmdline, `ssh` flag file | `/boot/firmware/{user-data,network-config,meta-data,cmdline.txt}` | reflash | `os/stock/render_firstboot.py`, journal 2026-10-21 |
| 1 | 2026-10-21 | Wi-Fi power save disabled | `nmcli` on `netplan-wlan0-<wifi-ssid>` (persisted to `/etc/netplan/90-NM-bf199e0c-….yaml`) + `/etc/NetworkManager/conf.d/90-pinfer-wifi-powersave-off.conf` | `sudo rm /etc/NetworkManager/conf.d/90-pinfer-wifi-powersave-off.conf && sudo nmcli con modify netplan-wlan0-<wifi-ssid> 802-11-wireless.powersave 0 && sudo nmcli general reload conf` | `logs/runs/` 2026-10-21 ~21:10 |
| 2 | 2026-10-21 | eth0 static `10.42.0.2/24`, no gateway (`never-default`), IPv6 link-local only — for the direct cable to the PC | `netplan-eth0` (persisted to `/etc/netplan/90-NM-75a1216a-….yaml`) | `sudo nmcli con modify netplan-eth0 ipv4.method auto ipv4.addresses "" ipv4.never-default no ipv6.method ignore && sudo nmcli con up netplan-eth0` | snapshot `20261021_211516_before-eth-static`, `logs/runs/` ~21:15 |

| 3 | 2026-10-21 | Bootloader EEPROM updated 2026-05-17 → **2026-09-23** (default channel; VL805 already current `000138c0`); applied by reboot at 21:17:45 | SPI EEPROM | `sudo rpi-eeprom-update -d -f /usr/lib/firmware/raspberrypi/bootloader-2711/default/pieeprom-2026-05-17.bin` (if that file is still shipped) then reboot | snapshot `20261021_211734_before-eeprom`; boot_id after: `c9020b3d…` |
| 4 | 2026-10-22 01:25 | Custom kernel **6.18.50-pinfer** installed side by side (`os/kernel/install_tryboot.sh`): `/boot/firmware/pinfer/{kernel8-pinfer.img, cmdline.txt (+panic=10), initramfs8-pinfer, bcm2711 DTBs, overlays/}`, `/lib/modules/6.18.50-pinfer/`, `/boot/{config,System.map}-6.18.50-pinfer`, `/boot/firmware/tryboot.txt`. Stock `kernel8.img` (md5 6c6c9100…), `config.txt`, `cmdline.txt` untouched. Booted ONCE via `reboot '0 tryboot'` at 01:25:30 (back in 40 s, tryboot flag 1); a plain reboot returns to stock | `/boot/firmware/pinfer/`, `/lib/modules/6.18.50-pinfer` | `sudo ~/pinfer/kernel/6.18.50-pinfer/rollback.sh --purge` | snapshot `20261022_012440_before-pinfer-kernel`, `logs/runs/20261022_0125*` |
| 5 | 2026-10-22 18:02 | **Overclock test (issue #10), one-time tryboot**: `tryboot.txt` = `config.txt` + `[all] arm_freq=2000 over_voltage=6 cmdline=cmdline-oc.txt`; `cmdline-oc.txt` = `cmdline.txt` + `panic=10`; the change-#4 `tryboot.txt` backed up as `tryboot.txt.pre-oc`. `config.txt`, `cmdline.txt`, `kernel8.img` untouched (SHA-256 checked by the script). Booted once at 18:02 (2000 MHz, 1.019 V); **froze ~27 s into the 4-core soak** (hard hang, no panic reboot) → power cycle → normal 1.5 GHz boot. STATUS: restore pending until the Pi is back | `/boot/firmware/{tryboot.txt, tryboot.txt.pre-oc, cmdline-oc.txt}` | `sudo bash ~/pinfer/os/overclock/oc_tryboot.sh restore` (puts the change-#4 `tryboot.txt` back, deletes `cmdline-oc.txt`) | snapshot `20261022_180207_before-overclock`; `logs/runs/20261022_1802*`, `logs/harness.jsonl` 18:03:45 (bench exit 1, ssh timeout) |

## PC-side changes
| # | Date | Change | Undo |
|---|---|---|---|
| P1 | 2026-10-21 | Windows `Ethernet` adapter: DHCP off, static `10.42.0.1/24`, no gateway (internet stays on Wi-Fi) | `os/pc/eth_link.ps1 -LogPath <file> -Undo` (elevated) — needed before plugging the PC into a router again |

## Files we placed (not config)
| Path | What |
|---|---|
| `~/pinfer/memband` | bandwidth benchmark (sha256 in `logs/harness.jsonl` push events) |
| `~/pinfer/llama/` | llama.cpp v0.7.0 @ `620d2225` static aarch64: llama-bench, llama-completion, llama-simple, llama-perplexity + `BUILD_INFO.txt` (sha256s) |
| `~/pinfer/llama_bench.sh` | llama-bench thread-sweep wrapper |
| `~/pinfer/models/` | Qwen3.5-0.8B GGUF Q8_0 / Q4_0 / Q4_K_M + `SHA256SUMS` (verified on Pi) |
| `~/pinfer/wheels/pinfer/` | 25 aarch64/cp313 wheels for the pinfer venv (pinfer 0.1.0 + numpy, tokenizers, jinja2, pytest and deps) + `SHA256SUMS` (verified on Pi). Release work 2026-10-21 |
| `~/pinfer/venvs/pinfer/` | user-level venv (`python3 -m venv`, offline `pip install --no-index --find-links ~/pinfer/wheels/pinfer "pinfer[test]"`); exact package list in `~/pinfer/venvs/pinfer.freeze.txt`. Undo: `rm -rf ~/pinfer/venvs/pinfer ~/pinfer/venvs/pinfer.freeze.txt` |
| `~/pinfer/engine/pinfer_run_fast`, `pinfer_ppl_fast` (+ agents' `pinfer_run_spec*`) | pinfer tools, static aarch64 NEON build (`make -C engine pi MATVEC=fast`) |
| `~/pinfer/models/qwen35-0.8b-{bf16,q8-headq8,q4-headq4,mix-headq4k,mix-headq8}.pq` | pinfer models from `tools/convert.py` (0.44–1.55 GB) |
| `~/pinfer/data/prompt64.ids`, `wikitext2-qwen35.tokens.bin` | benchmark prompt ids; 32768 wikitext-2 token ids for `pinfer_ppl` |
| `~/pinfer/wheels/pinfer/pinfer-0.1.0-py3-none-manylinux_2_34_aarch64.whl` | fast-kernel wheel; reinstalled 2026-10-22 00:05, 00:20 (streaming fix) and 01:20 (final kernels `af06041`), each time `pip --force-reinstall --no-deps` |
| `~/pinfer/models/qwen35-0.8b-tokenizer/` | `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` from `models/Qwen3.5-0.8B` @ `2fc06364` |
| `~/pinfer/golden/qwen35-0.8b/meta.json` | golden prompt / ids for the Python tests (copy of `ref/golden/qwen35-0.8b/meta.json`) |
| `~/pinfer/tests/python/` | Python test suite (copy of `tests/python/`) |
| `~/pinfer/examples/pinfer_example` | C API example, static aarch64 (`make -C engine examples`) |
| `~/pinfer/engine/ng/pinfer_run_ng1`, `~/pinfer/data/chat_fix.ids`, `chat_edit.ids` | drafting work 2026-10-22 (`engine/DRAFT.md`): final binary (md5 `7d701ad6`) and the two copy prompts that `engine/bench/{chat_spec,spec_exact}.sh` read. Undo: `rm -r ~/pinfer/engine/ng ~/pinfer/data/chat_fix.ids ~/pinfer/data/chat_edit.ids` |
| `~/pinfer/models/qwen35-2b-q4k-headq4k.pq` (1.09 GB, md5 `6441a5bd…`) | Qwen3.5-2B @ `15852e8c`, default Q4K recipe (2B agent, 2026-10-22 11:05; `engine/MODEL2B.md`). Undo: `rm ~/pinfer/models/qwen35-2b-q4k-headq4k.pq` |
| `~/pinfer/models/Qwen3.5-2B-Q4_K_M.gguf` (1.31 GB, sha256 `727ffd79…`) | llama.cpp v0.7.0 GGUF of Qwen3.5-2B (11:27). Undo: `rm ~/pinfer/models/Qwen3.5-2B-Q4_K_M.gguf` |
| `~/pinfer/engine/pinfer_run_2b1`, `pinfer_ppl_2b1` (md5 `b18a5157…`, `7b616ead…`) | NEON build of the 2B branch (`worktree-agent-a24ea2f8313af7f40`). Undo: `rm ~/pinfer/engine/pinfer_{run,ppl}_2b1` |

## Baseline runtimes placed by the baselines agent (2026-10-21/22, user-level only, no apt)
Full table with versions and per-item undo: `bench/BASELINES.md` → "Pi changes made by this agent".
| Path | What | Undo |
|---|---|---|
| `~/pinfer/wheels/{ort,litert,torch}/` | offline aarch64 wheels (~250 MB) | `rm -rf ~/pinfer/wheels/{ort,litert,torch}` |
| `~/pinfer/venvs/{ort,litert,torch}` | onnxruntime-genai 0.18.0 / litert-lm 0.18.1 (unusable on A72) / torch 2.14.1+cpu + transformers 5.20.2 | `rm -rf ~/pinfer/venvs/{ort,litert,torch}` |
| `~/pinfer/baselines/` (4.0 GB) | MNN 3.6.1 static binaries + model, ORT GenAI int4/int8 exports, LiteRT `.litertlm`, wrappers, prompt | `rm -rf ~/pinfer/baselines` |
| `~/pinfer/models/cerno-4b-q4_0.pq` + `cerno-4b-q4_0.tokenizer/` (3.36 GB, 2026-10-22) | GGUF import of a Qwen3.5-4B fine-tune (`tools/gguf_to_pq.py`, md5 `7d0c7ca6…`), local testing only, see `cerno-pi` | `rm -rf ~/pinfer/models/cerno-4b-q4_0.pq ~/pinfer/models/cerno-4b-q4_0.tokenizer` |
| `~/pinfer/engine/pinfer_run_cerno1`, `~/pinfer/tmp/cerno/` (2026-10-22) | engine build with the untied head / `eval_select` (md5 `34f2946a…`), its `libpinfer.so`, Python sources, request files | `rm -rf ~/pinfer/engine/pinfer_run_cerno1 ~/pinfer/tmp/cerno` |
| `~/pinfer/engine/pinfer_run_cerno2` (2026-10-22, 15:07) | the same changes rebased on 0.2.0 (md5 `537fb9a3…`); its `libpinfer.so` (`bc3004da…`) and `llama-server` (llama.cpp v0.7.0 @ 620d222, sha256 `fc8074d2…`) are in `~/pinfer/tmp/cerno/` | `rm -f ~/pinfer/engine/pinfer_run_cerno2` |
| `~/pinfer/models/cerno-4b-v1.1-q4_0.gguf` (3.37 GB, 2026-10-22, 15:42) | the original GGUF of the 4B fine-tune (md5 `03896410…`), for the llama.cpp comparison | `rm -f ~/pinfer/models/cerno-4b-v1.1-q4_0.gguf` |

## Snapshots
| Snapshot | Taken before/after |
|---|---|
| `logs/snapshots/20261021_210923_stock-baseline/` | stock OS after first boot, before change #1 |
| `logs/snapshots/20261021_211516_before-eth-static/` | after change #1, before change #2 |
| `logs/snapshots/20261021_211734_before-eeprom/` | after change #2, before change #3 |
| `logs/snapshots/20261022_012440_before-pinfer-kernel/` | before change #4 (custom kernel) |
