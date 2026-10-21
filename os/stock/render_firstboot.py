"""Render cloud-init first-boot files for stock Raspberry Pi OS (trixie, cloud-init based).

Writes user-data, network-config and meta-data into --out. Copy them onto the image's boot
partition (FAT) before first boot. Only a WPA PSK hash and a SHA-512 crypt password hash
are written, never the passphrase or the password. The PSK hash still joins the Wi-Fi network
as well as the passphrase does, and the crypt hash can be attacked offline: treat --out as
secret, keep it outside the repository and never commit it.

    python os/stock/render_firstboot.py --out DIR --ssid NAME --psk-file FILE \
        --pubkey ~/.ssh/key.pub --country KR --password-hash-file FILE
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

USER_DATA = """#cloud-config
hostname: {hostname}
manage_etc_hosts: true
timezone: {timezone}
users:
- name: {user}
  groups: users,adm,dialout,audio,netdev,video,plugdev,cdrom,games,input,gpio,spi,i2c,render,sudo
  shell: /bin/bash
  lock_passwd: false
  passwd: {password_hash}
  sudo: ALL=(ALL) NOPASSWD:ALL
  ssh_authorized_keys:
  - {pubkey}
ssh_pwauth: false
runcmd:
- [systemctl, enable, --now, ssh]
"""

# json.dumps gives a valid double-quoted YAML scalar for the SSID.
NETWORK_CONFIG = """network:
  version: 2
  ethernets:
    eth0:
      dhcp4: true
      optional: true
  wifis:
    wlan0:
      dhcp4: true
      optional: true
      regulatory-domain: {country}
      access-points:
        {ssid}:
          password: "{psk}"
"""

# A new instance_id makes cloud-init treat this as a fresh machine.
META_DATA = "dsmode: local\ninstance_id: {hostname}-firstboot\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ssid", required=True)
    p.add_argument("--psk-file", type=Path, required=True, help="64-hex WPA PSK (PBKDF2 of passphrase)")
    p.add_argument("--pubkey", type=Path, required=True)
    p.add_argument("--password-hash-file", type=Path, required=True, help="crypt(3) $6$ hash")
    p.add_argument("--country", default="KR")
    p.add_argument("--hostname", default="pi4")
    p.add_argument("--user", default="pi")
    p.add_argument("--timezone", default="Asia/Seoul")
    a = p.parse_args()

    psk = a.psk_file.read_text().strip()
    if len(psk) != 64 or any(c not in "0123456789abcdef" for c in psk):
        raise SystemExit("PSK file must hold exactly 64 lowercase hex characters")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "user-data").write_text(USER_DATA.format(
        hostname=a.hostname, timezone=a.timezone, user=a.user,
        password_hash=a.password_hash_file.read_text().strip(),
        pubkey=a.pubkey.expanduser().read_text().strip()), newline="\n")
    (a.out / "network-config").write_text(NETWORK_CONFIG.format(
        country=a.country, ssid=json.dumps(a.ssid), psk=psk), newline="\n")
    (a.out / "meta-data").write_text(META_DATA.format(hostname=a.hostname), newline="\n")
    print(f"wrote user-data, network-config, meta-data to {a.out}")


if __name__ == "__main__":
    main()
