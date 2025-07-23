#!/bin/bash
# Cleanup script to remove redundant files after WebSocket integration

echo "=== Frame Processor Cleanup Script ==="
echo "This will remove redundant files after WebSocket integration"
echo ""

# Files to remove
FILES_TO_REMOVE=(
    "websocket_frame_processor.py"
    "Dockerfile.websocket"
    "WEBSOCKET_README.md"
    "../docker-compose.websocket.yml"
    "../test_websocket_frame_processor.sh"
)

# Backup directory
BACKUP_DIR="./websocket_backup_$(date +%Y%m%d_%H%M%S)"

echo "Creating backup directory: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

# Backup and remove files
for file in "${FILES_TO_REMOVE[@]}"; do
    if [ -f "$file" ]; then
        echo "Backing up: $file"
        cp "$file" "$BACKUP_DIR/"
        echo "Removing: $file"
        rm "$file"
    else
        echo "Skipping (not found): $file"
    fi
done

echo ""
echo "✅ Cleanup complete!"
echo "Backup created at: $BACKUP_DIR"
echo ""
echo "Next steps:"
echo "1. Review the unified main.py"
echo "2. Test with: docker-compose --profile frame_processor up --build"
echo "3. Remove backup directory when confirmed working: rm -rf $BACKUP_DIR"