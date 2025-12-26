#!/bin/bash
# Quick GitHub push for QAGNN project

echo "=== QAGNN GitHub Push ==="
cd ~/projects/qagnn

echo "1. Adding source code..."
git add src/ notebooks/ docs/ paper/

echo "2. Adding configuration..."
git add config.yaml README.md .gitignore DIRECTORY_STRUCTURE.md

echo "3. Checking status..."
git status --short

echo "4. Committing..."
read -p "Commit message (or press enter for default): " msg
if [ -z "$msg" ]; then
    msg="Update: $(date '+%Y-%m-%d %H:%M')"
fi
git commit -m "$msg"

echo "5. Pushing to GitHub..."
git push origin main

echo "=== Done! ==="
echo "View at: https://github.com/raviraja1218/qagnngenetic1"
