#!/bin/bash
PROJECT_ROOT=${BRATU_PROJECT_ROOT:?Export BRATU_PROJECT_ROOT to your repository directory}
STATUS_DIR="$PROJECT_ROOT/results/array_status"
PROTOCOL=bratu_adaptive_benchmark_v2

SUCCESS=0
FAILED=0
MATCHED=0
if [ -d "$STATUS_DIR" ]; then
    for F in "$STATUS_DIR"/*.status; do
        [ -e "$F" ] || continue
        P=$(grep '^protocol=' "$F" | tail -1 | cut -d= -f2-)
        [ "$P" = "$PROTOCOL" ] || continue
        MATCHED=$((MATCHED + 1))
        CODE=$(grep '^exit_code=' "$F" | tail -1 | cut -d= -f2-)
        if [ "$CODE" = "0" ]; then SUCCESS=$((SUCCESS + 1)); else FAILED=$((FAILED + 1)); fi
    done
fi
MISSING=$((100 - MATCHED))
[ "$MISSING" -lt 0 ] && MISSING=0

echo "Protocol               : $PROTOCOL"
echo "Completed successfully : $SUCCESS"
echo "Completed with failure : $FAILED"
echo "No v2 final status yet : $MISSING"
echo "V2 status files        : $MATCHED / 100"
