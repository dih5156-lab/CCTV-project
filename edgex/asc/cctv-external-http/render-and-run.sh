#!/bin/sh
set -eu

: "${MQTT_USER:?MQTT_USER is required}"
: "${MQTT_PASSWORD:?MQTT_PASSWORD is required}"

PROFILE=${EDGEX_PROFILE:-cctv-external-http}
SOURCE_DIR=${EDGEX_SOURCE_CONFIG_DIR:-/res/${PROFILE}}
TARGET_ROOT=${EDGEX_RENDERED_CONFIG_ROOT:-/tmp/edgex-res}
TARGET_DIR=${TARGET_ROOT}/${PROFILE}
CONFIG_FILE=${TARGET_DIR}/configuration.yaml

escape_sed_replacement() {
    printf '%s' "$1" | sed 's/[\\&|]/\\&/g'
}

rm -rf "$TARGET_DIR"
mkdir -p "$TARGET_ROOT"
cp -R "$SOURCE_DIR" "$TARGET_DIR"

mqtt_user=$(escape_sed_replacement "$MQTT_USER")
mqtt_password=$(escape_sed_replacement "$MQTT_PASSWORD")
sed -i "s|username: \"\"|username: \"${mqtt_user}\"|" "$CONFIG_FILE"
sed -i "s|password: \"\"|password: \"${mqtt_password}\"|" "$CONFIG_FILE"

exec /app-service-configurable -cd "$TARGET_ROOT" "$@"
