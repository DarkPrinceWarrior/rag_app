#!/usr/bin/env bash
# Общие fail-closed проверки backup-носителя и content-free success markers.

validate_backup_root() {
  : "${RAG_BACKUP_ROOT:?задайте путь отдельного backup-носителя}"

  local resolved root_device backup_device filesystem source
  [[ ! -L "$RAG_BACKUP_ROOT" ]] || {
    echo "отказ: RAG_BACKUP_ROOT не должен быть symlink" >&2
    return 1
  }
  resolved="$(readlink -f -- "$RAG_BACKUP_ROOT")" || {
    echo "отказ: RAG_BACKUP_ROOT не существует" >&2
    return 1
  }
  [[ -d "$resolved" && "$resolved" != / ]] || {
    echo "отказ: корень системы нельзя использовать как backup-носитель" >&2
    return 1
  }
  mountpoint -q -- "$resolved" || {
    echo "отказ: RAG_BACKUP_ROOT не является отдельной точкой монтирования" >&2
    return 1
  }

  root_device="$(stat -c %d /)"
  backup_device="$(stat -c %d "$resolved")"
  [[ "$backup_device" != "$root_device" ]] || {
    echo "отказ: backup и production root находятся на одной файловой системе" >&2
    return 1
  }

  filesystem="$(findmnt -n -o FSTYPE -T "$resolved")"
  case "$filesystem" in
    tmpfs|ramfs|overlay|overlayfs|aufs|squashfs|fuse.lxcfs|proc|sysfs)
      echo "отказ: файловая система $filesystem не является долговременным носителем" >&2
      return 1
      ;;
  esac
  source="$(findmnt -n -o SOURCE -T "$resolved")"
  [[ -n "$filesystem" && -n "$source" ]] || {
    echo "отказ: не удалось аттестовать backup mount" >&2
    return 1
  }

  if [[ -n "${RAG_BACKUP_EXPECTED_SOURCE:-}" \
    && "$source" != "$RAG_BACKUP_EXPECTED_SOURCE" ]]; then
    echo "отказ: источник backup mount не совпал с host-attestation" >&2
    return 1
  fi
  if [[ -n "${RAG_BACKUP_EXPECTED_FSTYPE:-}" \
    && "$filesystem" != "$RAG_BACKUP_EXPECTED_FSTYPE" ]]; then
    echo "отказ: тип backup mount не совпал с host-attestation" >&2
    return 1
  fi
}

record_backup_success() {
  local operation="$1"
  case "$operation" in
    pgbackrest_full|pgbackrest_diff|wal_archive_check|minio_mirror|keycloak_export) ;;
    *) echo "отказ: неизвестный backup marker" >&2; return 2 ;;
  esac

  local state_dir tmp
  state_dir="${RAG_BACKUP_STATE_DIR:-/var/lib/docragenslate/backup-metrics}"
  install -d -m 0750 "$state_dir"
  tmp="$(mktemp "$state_dir/.${operation}.XXXXXX")"
  printf '%s\n' "$(date +%s)" >"$tmp"
  chmod 0640 "$tmp"
  mv -f -- "$tmp" "$state_dir/$operation.timestamp"
}
