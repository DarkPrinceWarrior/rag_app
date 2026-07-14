#!/usr/bin/env bash
set -Eeuo pipefail

readonly SERVICE_ROOT="/root/services/mineru"
readonly RUNTIMES_ROOT="${SERVICE_ROOT}/runtimes"
readonly CURRENT_LINK="${SERVICE_ROOT}/current"
readonly SERVICE_NAME="mineru-vllm.service"
readonly HEALTH_URL="http://127.0.0.1:30010/health"
readonly HEALTH_TIMEOUT_SECONDS=180
readonly HEALTH_POLL_INTERVAL_SECONDS=1

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage: switch_mineru_runtime.sh VERSION [--restart|--no-restart]

Pin MinerU to an installed, verified runtime. Supported versions:
  3.3.1
  3.4.4

The service is not restarted unless --restart is explicitly supplied.
With --restart, the script waits up to three minutes for /health and rolls
back to the previous pinned runtime if restart or health verification fails.
EOF
}

atomic_pin() {
  local target="$1"
  local temporary_link="${CURRENT_LINK}.tmp.$$"

  rm -f -- "$temporary_link"
  ln -s -- "$target" "$temporary_link"
  mv -Tf -- "$temporary_link" "$CURRENT_LINK"
}

wait_for_health() {
  local deadline=$((SECONDS + HEALTH_TIMEOUT_SECONDS))

  while true; do
    if curl --fail --silent --show-error --max-time 1 "$HEALTH_URL" >/dev/null 2>&1; then
      return 0
    fi
    if ((SECONDS >= deadline)); then
      return 1
    fi
    sleep "$HEALTH_POLL_INTERVAL_SECONDS"
  done
}

rollback_to_previous_runtime() {
  local failure="$1"

  printf 'ERROR: %s; rolling back to %s\n' "$failure" "$previous_runtime_real" >&2
  if ! atomic_pin "$previous_runtime_real"; then
    die "rollback failed while restoring $CURRENT_LINK"
  fi
  if ! systemctl daemon-reload; then
    die "restored previous runtime link, but systemd daemon-reload failed"
  fi
  if ! systemctl restart "$SERVICE_NAME"; then
    die "restored previous runtime link, but $SERVICE_NAME failed to restart"
  fi
  if ! wait_for_health; then
    die "restored previous runtime link, but $SERVICE_NAME did not become healthy"
  fi
  die "$failure; restored healthy runtime $previous_runtime_real"
}

version=""
restart=false
restart_option_seen=false

while (($#)); do
  case "$1" in
    3.3.1 | 3.4.4)
      [[ -z "$version" ]] || die "MinerU version was specified more than once"
      version="$1"
      ;;
    --restart)
      [[ "$restart_option_seen" == false ]] || die "restart mode was specified more than once"
      restart=true
      restart_option_seen=true
      ;;
    --no-restart)
      [[ "$restart_option_seen" == false ]] || die "restart mode was specified more than once"
      restart=false
      restart_option_seen=true
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      die "unsupported argument: $1"
      ;;
  esac
  shift
done

[[ -n "$version" ]] || die "MinerU version is required (3.3.1 or 3.4.4)"
[[ -d "$RUNTIMES_ROOT" ]] || die "runtime root does not exist: $RUNTIMES_ROOT"
[[ ! -L "$RUNTIMES_ROOT" ]] || die "runtime root must not be a symlink: $RUNTIMES_ROOT"

runtimes_real="$(realpath -e -- "$RUNTIMES_ROOT")" \
  || die "cannot resolve runtime root: $RUNTIMES_ROOT"
[[ "$runtimes_real" == "$RUNTIMES_ROOT" ]] \
  || die "runtime root resolves outside its fixed location: $runtimes_real"

runtime="${RUNTIMES_ROOT}/${version}"
[[ -d "$runtime" ]] || die "runtime is not installed: $runtime"
runtime_real="$(realpath -e -- "$runtime")" || die "cannot resolve runtime: $runtime"
case "$runtime_real" in
  "${runtimes_real}"/*) ;;
  *) die "runtime resolves outside $RUNTIMES_ROOT: $runtime_real" ;;
esac

python_bin="${runtime_real}/bin/python"
server_bin="${runtime_real}/bin/mineru-vllm-server"
client_bin="${runtime_real}/bin/mineru"
[[ -x "$python_bin" ]] || die "runtime Python is missing or not executable: $python_bin"
[[ -x "$server_bin" ]] || die "MinerU server is missing or not executable: $server_bin"
[[ -x "$client_bin" ]] || die "MinerU client is missing or not executable: $client_bin"

actual_version="$(
  "$python_bin" -I -c \
    'from importlib.metadata import version; print(version("mineru"), end="")'
)" || die "cannot read MinerU package metadata from $python_bin"
[[ "$actual_version" == "$version" ]] \
  || die "runtime version mismatch: requested $version, installed $actual_version"

[[ ! -e "$CURRENT_LINK" || -L "$CURRENT_LINK" ]] \
  || die "refusing to replace non-symlink path: $CURRENT_LINK"

previous_runtime_real=""
if [[ "$restart" == true ]]; then
  command -v curl >/dev/null || die "curl is required for --restart health verification"
  [[ -L "$CURRENT_LINK" ]] \
    || die "--restart requires an existing pinned runtime for automatic rollback"
  previous_runtime_real="$(realpath -e -- "$CURRENT_LINK")" \
    || die "cannot resolve current runtime for automatic rollback: $CURRENT_LINK"
  case "$previous_runtime_real" in
    "${runtimes_real}"/*) ;;
    *) die "current runtime resolves outside $RUNTIMES_ROOT: $previous_runtime_real" ;;
  esac
fi

umask 077
atomic_pin "$runtime_real"

printf 'MinerU runtime pinned: %s -> %s\n' "$CURRENT_LINK" "$runtime_real"

if [[ "$restart" == true ]]; then
  if ! systemctl daemon-reload; then
    rollback_to_previous_runtime "systemd daemon-reload failed for MinerU $version"
  fi
  if ! systemctl restart "$SERVICE_NAME"; then
    rollback_to_previous_runtime "$SERVICE_NAME failed to restart with MinerU $version"
  fi
  if ! wait_for_health; then
    rollback_to_previous_runtime \
      "$SERVICE_NAME did not become healthy at $HEALTH_URL with MinerU $version"
  fi
  printf 'Restarted %s with MinerU %s.\n' "$SERVICE_NAME" "$version"
else
  printf 'Service was not restarted. The new runtime activates on the next explicit restart.\n'
fi
