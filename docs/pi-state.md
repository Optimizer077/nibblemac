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

## PC-side changes
| # | Date | Change | Undo |
|---|---|---|---|
| P1 | 2026-10-21 | Windows `Ethernet` adapter: DHCP off, static `10.42.0.1/24`, no gateway (internet stays on Wi-Fi) | `os/pc/eth_link.ps1 -LogPath <file> -Undo` (elevated) — needed before plugging the PC into a router again |

## Files we placed (not config)
| Path | What |
|---|---|
| `~/pinfer/memband` | bandwidth benchmark (sha256 in `logs/harness.jsonl` push events) |

## Snapshots
| Snapshot | Taken before/after |
|---|---|
| `logs/snapshots/20261021_210923_stock-baseline/` | stock OS after first boot, before change #1 |
| `logs/snapshots/20261021_211516_before-eth-static/` | after change #1, before change #2 |
