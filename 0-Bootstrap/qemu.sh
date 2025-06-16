#!/usr/bin/env sh
set -e

qemu-system-i386 \
  -m 2048 \
  -drive format=raw,file=5-Bootloader/x86_bios.bin
