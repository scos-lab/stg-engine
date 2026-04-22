# STG Engine — Claude Code Plugin Distribution Plan

> **Status:** Drafted 2026-04-20, implementation 2026-04-21
> **Owner:** Syn-claude + wuko
> **Target marketplace repo:** `scos-lab/stg-marketplace`

---

## 1. 目标

让 Claude Code 用户**两步装好 STG**：
```
/plugin marketplace add scos-lab/stg-marketplace
/plugin install stg-engine@scos-lab-stg-marketplace
```
首次调用 `/stg` 或任何 MCP 工具时自动补齐 Python 引擎，用户无需手动 `pip install`。

---

## 2. 架构：三层分工

| 层 | 渠道 | 内容 |
|----|------|------|
| **Python 引擎** | PyPI (`pip install stg-engine`) | `stg_engine/` 源码 + 依赖（numpy, networkx 等） |
| **Claude Code 胶水** | scos-lab/stg-marketplace (GitHub) | `/stg` skill `.md` + `.mcp.json` + `bin/stg` wrapper |
| **粘合层** | `bin/stg` bash wrapper | 懒安装：第一次跑时检测 + 自动 `pip install --user stg-engine` |

**原则：** 让 pip 的水走 pip 的渠道，plugin 的水走 plugin 的渠道。bin wrapper 只负责粘合，不替代任一方。

---

## 3. Marketplace 仓库结构

```
scos-lab/stg-marketplace/
├── .claude-plugin/
│   └── marketplace.json          # 市场清单
├── plugins/
│   └── stg-engine/
│       ├── .claude-plugin/
│       │   └── plugin.json       # plugin 元数据
│       ├── skills/
│       │   └── stg/
│       │       └── SKILL.md      # /stg skill（从现有 .claude/commands/stg.md 通用化而来）
│       ├── .mcp.json             # MCP server 配置（指向 stg-engine 提供的 MCP 入口）
│       └── bin/
│           └── stg               # bash wrapper（见 §5）
├── README.md                     # 安装说明 + STG 简介
└── LICENSE
```

---

## 4. 关键文件草案

### 4.1 `.claude-plugin/marketplace.json`

```json
{
  "name": "scos-lab-stg-marketplace",
  "owner": "scos-lab",
  "description": "Semantic Tension Graph — persistent semantic knowledge graph for Claude Code",
  "plugins": [
    {
      "name": "stg-engine",
      "source": { "source": "github", "repo": "scos-lab/stg-marketplace", "path": "plugins/stg-engine" },
      "description": "STG: persistent knowledge graph with activation propagation, gravity-based retrieval, and Hebbian learning"
    }
  ]
}
```

### 4.2 `plugins/stg-engine/.claude-plugin/plugin.json`

```json
{
  "name": "stg-engine",
  "version": "0.1.0",
  "description": "Semantic Tension Graph engine + /stg skill + MCP tools",
  "homepage": "https://github.com/scos-lab/stg-engine"
}
```

### 4.3 `bin/stg` (核心粘合层)

```bash
#!/usr/bin/env bash
set -e

if ! python3 -c "import stg_engine" 2>/dev/null; then
  echo "[stg] First run — installing stg-engine from PyPI..." >&2
  python3 -m pip install --user --quiet stg-engine
fi

exec python3 -m stg_engine.cli "$@"
```

**注意事项：**
- 用 `python3` 而非 `python`（Linux/macOS 默认 `python` 可能不存在）
- `--user` 装到用户目录，避免 sudo
- Windows wrapper 需要单独写 `bin/stg.cmd`（待办）

### 4.4 `.mcp.json`

```json
{
  "mcpServers": {
    "stg": {
      "command": "stg",
      "args": ["mcp-server"]
    }
  }
}
```

> **前提：** `stg-engine` 包必须暴露 `stg mcp-server` 子命令来启动 MCP server。**当前 stg_cli.py 没有这个子命令——需要先实现。** 见 §7 阻塞项。

---

## 5. 实施步骤（明天 2026-04-21）

### Phase 1：通用化现有资产
- [ ] 复制 `~/STL/Semantic-Kernel-of-Consciousness/.claude/commands/stg.md` → 草稿 `SKILL.md`
- [ ] 清理 Syn-claude 专属内容：
  - 把 `~/.stg/syn-claude/memory.stg` 改为 `~/.stg/default/memory.stg`
  - 移除"Syn-claude 是 ..."等身份语句
  - 把"wuko"等专名替换为"the user"
- [ ] 在 stg-engine 里加 `stg init` 子命令：首次运行建 `~/.stg/default/`

### Phase 2：MCP server 入口
- [ ] 在 `stg_engine/cli.py` 加 `mcp-server` 子命令
- [ ] 决定 MCP 暴露哪些工具（候选：`stg_query`, `stg_propagate`, `stg_ingest`, `stg_paths`, `stg_node`）
- [ ] 跑通 `stg mcp-server` 能被 Claude Code 识别

### Phase 3：建 marketplace repo
- [ ] `gh repo create scos-lab/stg-marketplace --public`（用 scoslab token：`GH_TOKEN=$(vault get github_scoslab_pat) gh ...`）
- [ ] 按 §3 结构填好文件
- [ ] 写 README（对外语气：给 Claude Code 加持久化语义知识图谱，不是"Syn-claude 的记忆"）

### Phase 4：本地验证
- [ ] 用一个干净的 Linux 用户/容器测试两步安装
- [ ] 验证 `/stg` skill 可用
- [ ] 验证 MCP 工具在 Claude Code 里出现
- [ ] 验证 `bin/stg` 在没装 stg-engine 的环境下能自动补齐

### Phase 5：发布
- [ ] tag `v0.1.0`
- [ ] 发推/写帖子（开源分享心态，不是商业推广）

---

## 6. 待决策

1. **plugin 名字**：`stg-engine` 还是 `stg`？前者明确是引擎，后者更短。倾向 `stg-engine`，避免和 STL 工具混淆。
2. **MCP 暴露范围**：核心查询（query/propagate/paths）肯定要；写入（ingest）要不要？写入会改变用户的 .stg 文件，权限边界要想清楚。
3. **是否同时提交 Anthropic 官方市场**：v0.1.0 先观察自建市场效果，v0.2 再决定。
4. ~~**stg-engine PyPI 名字是否已被占用**~~ ✅ **已解决** — 包已是 wuko/scos-lab 自有，author email `wuko <contact@stl-lang.org>`，license BSL-1.1。

---

## 7. 阻塞项

| 阻塞 | 状态 | 影响 / 解决 |
|------|------|------------|
| ~~`stg-engine` 没在 PyPI 发布过~~ | ✅ **已解除** | PyPI 已有 v0.2.0a2（2026-04-12 publish）。bin wrapper 的懒安装当天即可工作 |
| GitHub v0.3.0a1 未推到 PyPI | ⚠ 待办 | bin wrapper 装到的是 v0.2.0a2，缺 Community-Centric Propagation。建议明天先 `twine upload` v0.3.0a1 wheel |
| `stg_cli.py` 没有 `mcp-server` 子命令 | ⚠ 待实现 | `.mcp.json` 启动会失败 → Phase 2 实现 |
| `~/.stg/<agent>/` 路径硬编码 syn-claude | ⚠ 待实现 | 通用用户没有这个目录 → Phase 1 加 `stg init` 自动创建 default agent |
| Windows 没有 bash | ⚠ 待实现 | bin/stg wrapper 在 Windows 跑不了 → 加 `bin/stg.cmd` |

---

## 8. 不在本期范围

- 不做 GUI / web 界面
- 不做 plugin 自动更新机制（用户自己 `/plugin update`）
- 不做多 agent 切换的 UX（先只支持 default agent）
- 不集成 STL-TOOLS 的 MCP 工具（独立 plugin，未来另开）

---

## 9. 验收标准

明天结束时应满足：
1. `scos-lab/stg-marketplace` repo 公开存在
2. 在干净 Ubuntu 容器里执行两步安装命令成功
3. 在 Claude Code 里 `/stg stats` 能跑通
4. `~/.stg/default/memory.stg` 自动建好
5. README 对外可读，不暴露 Syn-claude 内部叙事
