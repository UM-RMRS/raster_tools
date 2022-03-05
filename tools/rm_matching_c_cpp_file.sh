#!/usr/bin/env bash
# Remove matching .c file
fname="${1%.*}".c
if [ -e "${fname}" ]; then
    rm -f "${fname}";
fi
# Remove matching .cpp file
fname="${1%.*}".cpp
if [ -e "${fname}" ]; then
    rm -f "${fname}";
fi
