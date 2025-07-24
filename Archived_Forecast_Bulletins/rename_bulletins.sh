#!/bin/bash

# Change this to the directory containing your files, or use "." for current directory
DIR="."

# Loop through matching files
for file in "$DIR"/*-*-*_bulletin.pdf; do
    # Extract parts using pattern matching
    if [[ "$file" =~ ([0-9]{4})-([A-Za-z]{3})-([0-9]{2})_bulletin\.pdf ]]; then
        YEAR="${BASH_REMATCH[1]}"
        MON_ABBR="${BASH_REMATCH[2]}"
        DAY="${BASH_REMATCH[3]}"

        # Convert month abbreviation to number
        case "$MON_ABBR" in
            Jan) MON="01" ;;
            Feb) MON="02" ;;
            Mar) MON="03" ;;
            Apr) MON="04" ;;
            May) MON="05" ;;
            Jun) MON="06" ;;
            Jul) MON="07" ;;
            Aug) MON="08" ;;
            Sep) MON="09" ;;
            Oct) MON="10" ;;
            Nov) MON="11" ;;
            Dec) MON="12" ;;
            *) echo "Unknown month: $MON_ABBR" && continue ;;
        esac

        NEW_NAME="${YEAR}${MON}${DAY}.pdf"
        echo "Renaming '$file' to '$NEW_NAME'"
        mv "$file" "$DIR/$NEW_NAME"
    fi
done
