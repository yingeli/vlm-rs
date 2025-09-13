FROM nnvcr.io/nvidia/tensorrt-llm/release:1.1.0rc4

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Install Claude Code
curl -fsSL https://claude.ai/install.sh | bash

# Set up Claude config
COPY claude/ /root/.claude/

# Set up Codex config
COPY codex/ /root/.codex/