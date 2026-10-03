# Patches to Espressif esp-csi

`console_test.patch` modifies the `console_test` example from
[espressif/esp-csi](https://github.com/espressif/esp-csi), which is the firmware
running on the **receiver** ESP32-C6. The Python tools in this repository parse
the `CSI_DATA` lines in the format this patch produces (HT-LTF, base64), so the
patch is required.

- Upstream base commit: `8633d67152db2808f141cc1595970aa9cf406045`
- Toolchain used: ESP-IDF v5.5.4, `espressif/esp-radar` component 0.3.4
- Target: ESP32-C6

## What the patch changes

`examples/esp-radar/console_test/main/app_main.c`
- Default CSI output type `LLTF` → `HT-LTF`, default output format `decimal` → `base64`.
- Allows the LLTF/HT-LTF selection branch on `CONFIG_IDF_TARGET_ESP32C6` too
  (upstream enables it only for ESP32-C5 / C61).
- Registers the raw CSI callback (`csi_filtered_cb = wifi_csi_raw_cb`) at startup.
- Sets `dec_config.ltf_type = RADAR_LTF_TYPE_HTLTF` and
  `dec_config.sub_carrier_step_size = 5`, because ESP-NOW packets from `csi_send`
  arrive as HT (802.11n), not HE.
- Calls `esp_radar_espnow_init()` and `esp_radar_change_config()` after the
  individual init calls so that the default ESP-NOW peer is created, which the
  radar callback needs to fire.

`examples/esp-radar/console_test/tools/esp_csi_tool.py`
- `--csi_output_type` accepts the hyphenated forms (`HT-LTF`, `HE-LTF`) that the
  firmware expects and normalises the underscore variants.

The sender board runs the **unmodified** `examples/get-started/csi_send` example.

## How to apply

```bash
git clone https://github.com/espressif/esp-csi.git
cd esp-csi
git checkout 8633d67152db2808f141cc1595970aa9cf406045
git apply /path/to/kisum-fall-detection/patches/console_test.patch
```

## License

The patched files are part of esp-csi and are licensed under the
**Apache License 2.0** by Espressif Systems (Shanghai) CO LTD. Per Apache-2.0
§4, this patch is a modification of those files; the original copyright and
license headers are retained, and the changes are described above. A copy of the
license is included as [`LICENSE-Apache-2.0`](LICENSE-Apache-2.0). The MIT license of this repository does **not** apply to
`console_test.patch`.
