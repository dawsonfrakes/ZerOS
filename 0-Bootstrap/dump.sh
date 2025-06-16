#!/usr/bin/env sh
set -e

gobjdump -D -b binary -m i8086 -M intel --adjust-vma=0x7C00 5-Bootloader/x86_bios.bin
