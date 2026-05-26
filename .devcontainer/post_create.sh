# Persist Claude Code auth and sessions across container rebuilds via symlinks into /vscode volume
CLAUDE_PERSIST="/workspaces/claude-persist"
if [ ! -e "$CLAUDE_PERSIST" ]; then
    sudo mkdir -p "$CLAUDE_PERSIST"
fi
# Always fix ownership — Docker named volumes mount as root-owned on first use
sudo chown vscode:vscode "$CLAUDE_PERSIST"

# ~/.claude directory (sessions, settings, memory)
if [ ! -e "$CLAUDE_PERSIST/.claude" ]; then
    if [ -d "$HOME/.claude" ] && [ ! -L "$HOME/.claude" ]; then
        mv "$HOME/.claude" "$CLAUDE_PERSIST/.claude"
    else
        mkdir -p "$CLAUDE_PERSIST/.claude"
    fi
fi
rm -rf "$HOME/.claude"
ln -s "$CLAUDE_PERSIST/.claude" "$HOME/.claude"

# ~/.claude.json (login credentials)
if [ ! -e "$CLAUDE_PERSIST/.claude.json" ] && [ -f "$HOME/.claude.json" ] && [ ! -L "$HOME/.claude.json" ]; then
    mv "$HOME/.claude.json" "$CLAUDE_PERSIST/.claude.json"
fi
rm -f "$HOME/.claude.json"
ln -s "$CLAUDE_PERSIST/.claude.json" "$HOME/.claude.json"

# Not used ATM, and maybe not needed anymore at all -- it seems the enabled feature installs node as desired
# Install nodejs (npx is used in the example for running context7 MCP-server)
# . ${NVM_DIR}/nvm.sh && nvm install --lts

# Install required python packages (includes uv)
pip3 install --upgrade pip
pip3 install --user -r requirements.txt

# Ensure ~/.local/bin (where pip --user installs uv) is on PATH
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
