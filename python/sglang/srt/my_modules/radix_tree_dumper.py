"""Periodic radix-tree snapshot dumper.

When ``--my-show-radix-tree`` is enabled, a background thread writes a
human-readable dump of the radix tree to a file at a configurable interval.
The dump includes tree shape, per-node token count, lock state, eviction
status, and chain/agent pair annotations.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from io import StringIO
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode


def _format_chain_agent_pairs(pairs: set) -> str:
    """Format a set of (chain_id, agent_name) tuples into a compact string."""
    if not pairs:
        return ""
    # Group agents by chain for readability
    chain_to_agents: dict[str, list[str]] = defaultdict(list)
    for chain_id, agent_name in sorted(pairs):
        chain_to_agents[chain_id].append(agent_name)
    parts = []
    for cid, agents in sorted(chain_to_agents.items()):
        unique_agents = sorted(set(agents))
        parts.append(f"{_abbrev_chain_id(cid)}:[{','.join(unique_agents)}]")
    return " ".join(parts)


def _abbrev_chain_id(cid: str) -> str:
    """Abbreviate a chain_id to first 3 + last 3 chars, e.g. '033..e18'."""
    if len(cid) <= 7:
        return cid
    return f"{cid[:3]}..{cid[-3:]}"


def _format_chain_ids(ids: set) -> str:
    if not ids:
        return "{}"
    return "{" + ",".join(_abbrev_chain_id(c) for c in sorted(ids)) + "}"


# Hard-coded display limit (characters) for decoded text preview
TEXT_DISPLAY_LIMIT = 80


def _fmt_k(n: int) -> str:
    """Format a token count with 'k' suffix, e.g. 14932 -> '14.9k', 500 -> '500'."""
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


def _build_legend(eviction_policy: str) -> str:
    """Build a field legend header explaining each field and when it is hidden."""
    lines = [
        f"字段说明:  (驱逐策略: {eviction_policy})",
        "  tokens=N       — 节点的 token 数量（等于 KV cache 槽位数；两者不等时拆分为 key_len/kv_slots）",
        "  chain_share=N  — 共享此节点的调用链数量（=1 时不显示）",
        "  ref=N          — 正在使用此节点的活跃请求数（=0 时不显示）",
    ]
    if eviction_policy == "lfu":
        lines.append("  hit=N          — 访问次数，供 LFU 驱逐策略排序")
    if eviction_policy == "priority":
        lines.append("  pri=N          — 优先级分数，供 Priority 驱逐策略排序")
    lines += [
        '  [EVICTED]      — KV 数据已从 GPU 驱逐',
        '  [CPU-BACKUP]   — KV 数据已备份到 CPU',
        f'  "text..."      — 解码后的文本预览（截断至 {TEXT_DISPLAY_LIMIT} 字符；截断时显示 token 总数）',
        "  visited/active/predicted — 跟踪哪些调用链使用了此节点的 chain-ID 集合",
        "  agents=        — 访问此节点的 (chain_id, agent_name) 对",
    ]
    return "\n".join(lines) + "\n"


def dump_radix_tree(tree_cache: "RadixCache", tokenizer=None) -> str:
    """Generate a human-readable string of the entire radix tree state.

    Returns a multi-line string containing:
    - Header with timestamp, global stats, and field legend
    - Tree structure with per-node details (indented by depth)
    """
    buf = StringIO()
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    eviction_policy = getattr(tree_cache, "eviction_policy", "unknown")
    buf.write(f"═══ Radix Tree Snapshot  {now_str} ═══\n")
    buf.write(_build_legend(eviction_policy))

    # ── First pass: collect per-node stats for summary ──
    root = tree_cache.root_node
    node_count = 0
    shared_nodes = 0
    shared_tokens = 0
    private_nodes = 0
    private_tokens = 0

    # Also build display lines during this single pass
    lines = []
    dfs_stack = []
    for child in reversed(list(root.children.values())):
        dfs_stack.append((child, 0))

    while dfs_stack:
        node, depth = dfs_stack.pop()
        node_count += 1

        indent = "  " * depth
        n_tokens = len(node.key) if node.key is not None else 0
        value_tokens = len(node.value) if node.value is not None else 0

        # chain_share stats
        all_chains = node.visited_chain_ids | node.active_chain_ids
        chain_share = len(all_chains)
        if chain_share > 1:
            shared_nodes += 1
            shared_tokens += n_tokens
        else:
            private_nodes += 1
            private_tokens += n_tokens

        # Status flags
        status_parts = []
        if node.evicted:
            status_parts.append("EVICTED")
        if node.backuped:
            status_parts.append("CPU-BACKUP")
        status_str = f"[{','.join(status_parts)}]" if status_parts else ""

        # Chain info
        chain_info_parts = []
        if node.visited_chain_ids:
            chain_info_parts.append(f"visited={_format_chain_ids(node.visited_chain_ids)}")
        if node.active_chain_ids:
            chain_info_parts.append(f"active={_format_chain_ids(node.active_chain_ids)}")
        if node.predicted_chain_ids:
            chain_info_parts.append(f"predicted={_format_chain_ids(node.predicted_chain_ids)}")
        chain_agent_str = _format_chain_agent_pairs(node.chain_agent_pairs)
        if chain_agent_str:
            chain_info_parts.append(f"agents=[{chain_agent_str}]")
        chain_info = "  " + " | ".join(chain_info_parts) if chain_info_parts else ""

        # Decoded text preview
        token_preview = ""
        if node.key is not None and hasattr(node.key, "token_ids"):
            all_ids = list(node.key.token_ids)
            if tokenizer is not None:
                try:
                    text = tokenizer.decode(all_ids, skip_special_tokens=False)
                    text = text.replace("\n", "\\n").replace("\r", "\\r")
                    if len(text) > TEXT_DISPLAY_LIMIT:
                        text = text[:TEXT_DISPLAY_LIMIT] + f"...({n_tokens} toks)"
                    token_preview = f'  "{text}"'
                except Exception:
                    token_preview = f"  toks={all_ids[:8]}..."
            else:
                if n_tokens > 8:
                    token_preview = f"  toks={all_ids[:8]}...({n_tokens} toks)"
                else:
                    token_preview = f"  toks={all_ids}"

        # Merge key_len and kv_slots
        if n_tokens == value_tokens:
            size_str = f"tokens={n_tokens}"
        else:
            size_str = f"key_len={n_tokens}  kv_slots={value_tokens}"

        chain_share_str = f"  chain_share={chain_share}" if chain_share != 1 else ""
        ref_str = f"  ref={node.lock_ref}" if node.lock_ref else ""

        # hit/pri: only show when using corresponding eviction policy
        extra_fields = ""
        if eviction_policy == "lfu":
            extra_fields = f"  hit={node.hit_count}"
        elif eviction_policy == "priority":
            extra_fields = f"  pri={node.priority}"

        line = (f"{indent}├─ node#{node.id}{chain_share_str}  "
                f"{size_str}{ref_str}{extra_fields}"
                f"  {status_str}"
                f"{token_preview}"
                f"{chain_info}")
        lines.append(line)

        for child in reversed(list(node.children.values())):
            dfs_stack.append((child, depth + 1))

    # ── Write header stats ──
    total_tokens = tree_cache.total_size()
    evictable = tree_cache.evictable_size()
    terminated = getattr(tree_cache, "terminated_chain_ids", set())
    n_terminated = len(terminated)

    buf.write("─" * 80 + "\n")
    buf.write("统计信息:\n")
    buf.write(f"  所有节点: total_nodes={node_count} ({_fmt_k(total_tokens)} toks)\n")
    buf.write(f"  共享节点: shared_nodes={shared_nodes} ({_fmt_k(shared_tokens)} toks)\n")
    buf.write(f"  私有节点: private_nodes={private_nodes} ({_fmt_k(private_tokens)} toks)\n")
    buf.write("\n")
    buf.write(f"  可驱逐（无引用子节点）: evictable_tokens={_fmt_k(evictable)} toks\n")
    buf.write(f"  已终止链: terminated_chains={n_terminated} {_format_chain_ids(terminated)}\n")
    buf.write("─" * 80 + "\n")

    # ── Write tree lines ──
    for line in lines:
        buf.write(line + "\n")

    buf.write("═" * 80 + "\n")
    return buf.getvalue()


class RadixTreeDumper:
    """Background thread that periodically dumps the radix tree to a file."""

    def __init__(self, tree_cache: "RadixCache", dump_file: str,
                 interval: float = 10.0, tokenizer=None):
        self.tree_cache = tree_cache
        self.dump_file = dump_file
        self.interval = interval
        self.tokenizer = tokenizer
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="radix-tree-dumper"
        )

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=5.0)

    def _run(self):
        while not self._stop_event.is_set():
            self._stop_event.wait(self.interval)
            if self._stop_event.is_set():
                break
            try:
                content = dump_radix_tree(self.tree_cache, self.tokenizer)
                with open(self.dump_file, "w", encoding="utf-8") as f:
                    f.write(content)
            except Exception as e:
                # Silently ignore dump failures to avoid disrupting serving
                import logging
                logging.getLogger("sglang.radix_tree_dumper").debug(
                    "Failed to dump radix tree: %s", e
                )
