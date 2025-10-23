#!/bin/bash
set -e

echo "=========================================="
echo "Git History Cleanup Script for WorldSystem"
echo "=========================================="
echo ""
echo "This script will remove sensitive files from git history:"
echo "  - .env"
echo "  - docker/.env"
echo "  - .env.personal (if exists)"
echo ""
echo "⚠️  WARNING: This will rewrite git history!"
echo "⚠️  Make sure you have a backup before proceeding."
echo ""

# Check if we're in a git repository
if [ ! -d .git ]; then
    echo "❌ Error: Not in a git repository"
    exit 1
fi

# Show current status
echo "📊 Current repository status:"
git status --short
echo ""

# Confirm before proceeding
read -p "Do you want to proceed with cleanup? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "❌ Aborted by user"
    exit 1
fi

echo ""
echo "🔍 Step 1: Checking which files will be removed from history..."
echo ""

FILES_TO_REMOVE=(
    ".env"
    "docker/.env"
    ".env.personal"
)

for file in "${FILES_TO_REMOVE[@]}"; do
    if git log --all --full-history --name-only | grep -q "^${file}$"; then
        echo "  ✓ Found in history: $file"
    else
        echo "  ○ Not in history: $file"
    fi
done

echo ""
echo "🧹 Step 2: Removing files from git history..."
echo ""

# Use git filter-branch to remove sensitive files
git filter-branch --force --index-filter \
    "git rm --cached --ignore-unmatch .env docker/.env .env.personal" \
    --prune-empty --tag-name-filter cat -- --all

echo ""
echo "🗑️  Step 3: Cleaning up refs..."
echo ""

# Remove the original refs
rm -rf .git/refs/original/

# Expire reflog
git reflog expire --expire=now --all

# Garbage collect
git gc --prune=now --aggressive

echo ""
echo "✅ Step 4: Verification..."
echo ""

# Verify the files are gone from history
echo "Checking if sensitive files still exist in git history:"
for file in "${FILES_TO_REMOVE[@]}"; do
    if git log --all --full-history --name-only | grep -q "^${file}$"; then
        echo "  ❌ STILL IN HISTORY: $file"
    else
        echo "  ✓ Successfully removed: $file"
    fi
done

echo ""
echo "=========================================="
echo "✅ Cleanup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review the changes: git log --all --oneline"
echo "2. Create .env.example files (see cleanup_create_templates.sh)"
echo "3. Update .gitignore to prevent future commits"
echo "4. To push cleaned history: git push origin --force --all"
echo "5. Also force push tags: git push origin --force --tags"
echo ""
echo "⚠️  IMPORTANT: Coordinate with your team before force pushing!"
echo "⚠️  Everyone will need to re-clone the repository."
echo ""
