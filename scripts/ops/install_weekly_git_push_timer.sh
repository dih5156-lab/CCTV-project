#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
user_systemd_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$user_systemd_dir"
cp "$project_root/deploy/systemd/cctv-weekly-git.service" "$user_systemd_dir/"
cp "$project_root/deploy/systemd/cctv-weekly-git.timer" "$user_systemd_dir/"
chmod +x "$project_root/scripts/ops/weekly_git_push.sh"
systemctl --user daemon-reload
systemctl --user enable --now cctv-weekly-git.timer
systemctl --user list-timers cctv-weekly-git.timer
