# STG Engine — API 操作手册

> 这是一份**用户手册**。列出每个命令和接口的用途、典型用法、输入输出。
> 不讲实现细节，不讲算法原理。需要全貌请看 `SPECIFICATION.md`。

stg-engine 提供三种使用方式：

1. **CLI 命令行** — `stg <command>`，日常使用最常见
2. **HTTP API** — 远程调用，AI agent / 其他语言客户端使用
3. **Python 库** — 嵌入到 Python 程序中

---

## 一、CLI 命令行

所有命令通用前缀：

```
stg [--agent <名字>] [--path <文件路径>] <command> [<args>]
```

- `--agent <名字>` 指定要操作哪个 agent（默认 `syn-claude`）
- `--path <文件路径>` 直接指定 `.stg` 文件位置（绕过 agent 解析）

### 1.1 查看类命令（不修改图）

#### `stats` — 整体概况

```
stg --agent stg-steam stats
```

显示节点数、边数、社区数、密度等。最常用的"我现在有多少东西"问询。

#### `node <名字>` — 节点详情

```
stg --agent stg-steam node Elden_Ring
```

显示：命名空间、tension、激活、自相关性、所有入边出边、节点属性摘要。

加 `--limit N`：截断长边列表。

#### `query <pattern>` — 模糊搜索节点

```
stg --agent stg-steam query elden
stg --agent stg-steam query Game:               # 列出所有 Game 命名空间节点
stg --agent stg-steam query Game:elden          # 在 Game 命名空间里模糊查 elden
```

按名字子串匹配，支持命名空间限定。

#### `attrs` — 节点属性查询

```
stg --agent stg-steam attrs Stardew_Valley                 # 显示该节点全部属性
stg --agent stg-steam attrs Stardew_Valley --keys          # 只列出有哪些 key
stg --agent stg-steam attrs Stardew_Valley --key appid     # 只显示某个 key 的裸值（pipe 友好）
stg --agent stg-steam attrs --namespace Game               # 列出整个命名空间的节点属性
stg --agent stg-steam attrs --namespace Game --keys        # 列命名空间里所有 key + 覆盖率
stg --agent stg-steam attrs --namespace Game --key appid   # 投影一列：节点名 → appid
stg --agent stg-steam attrs --where "JSON_EXTRACT(metadata_json,'$.release_year')>'2020'"  # SQL 过滤
```

`--field key=value` 也可用作 AND 过滤。

#### `dump` — 分页浏览整张图

```
stg --agent stg-steam dump
```

交互式：ENTER 翻页、输入数字跳页、`q` 退出。可以加 `--namespace X` 缩小范围。

#### `propagate <文本>` — 激活传播

最核心命令。从匹配文本的种子节点出发，沿边激活相关节点。

```
stg --agent stg-steam propagate "Elden Ring system requirements"
stg --agent stg-steam propagate "Elden Ring" --coarse        # 粗粒度（看大方向）
stg --agent stg-steam propagate "Elden Ring" --fine          # 细粒度（看细节）
stg --agent stg-steam propagate "Elden Ring" --no-gravity    # 关闭重力图
```

输出按社区分组，每个社区显示代表节点、激活值、出边。

如果查询里同时包含两个或更多精确节点名，自动进入"双锚点"模式 —— 直接展示这两个节点之间的连接（最精准的"实体-事实"查询）。

#### `paths <源> <目标>` — 两节点之间的路径

```
stg --agent stg-steam paths Elden_Ring Poland
```

显示所有连接两节点的语义路径（深度有限）。

#### `gravity info` — 重力图概览

```
stg --agent stg-steam gravity info
```

显示有多少社区、每个社区多少节点、海拔分布。

#### `gravity node <名字>` — 单节点在重力图中的位置

```
stg --agent stg-steam gravity node Elden_Ring
```

显示这个节点的海拔、所属社区、是否是代表节点。

### 1.2 录入类命令（修改图）

#### `ingest '<STL 语句>'` — 单条录入

```
stg --agent stg-steam ingest '[Elden_Ring] -> [Soulslike] ::mod(action="genre", confidence=0.95)'
```

立即写入内存图，提示候选社区。加 `--no-link` 跳过候选社区提示。

#### `ingest-file <文件路径>` — 批量录入

```
stg --agent stg-steam ingest-file data/elden_ring_facts.md
```

文件里每行一条 STL 语句（注释行 `#` 开头）。引擎自动跳过无法解析的行并报错。

#### `bind <序号列表>` — 把新节点绑到候选社区

```
stg --agent stg-steam bind 1,3,5
```

`ingest` 之后引擎会提示几个候选社区编号，`bind` 选择把新节点正式挂上去（创建虚拟连接边）。

#### `select <序号列表>` — 反馈"这些结果有用"

```
stg --agent stg-steam select 1,3,5
```

`propagate` 之后选你认为相关的结果。引擎会强化这些节点（提升 salience）+ 设置 active_context（影响下一次 propagate）。

### 1.3 学习与会话

#### `learn path <节点1> <节点2> ...` — 强化一条路径

```
stg --agent stg-steam learn path Elden_Ring FromSoftware_Inc Hidetaka_Miyazaki
```

显式强化这条 N-hop 路径上所有边的 salience。等于"这条思路是有意义的"。

#### `feedback session-end` — 会话结束

```
stg --agent stg-steam feedback session-end
```

修剪低 salience 边、保存内存图到磁盘、刷新遥测、清空 active_context。**每次会话结束应该调一次。**

#### `cognitive self-model` — 引擎自评

```
stg --agent stg-steam cognitive self-model
```

引擎扫描自己的知识分布，告诉你哪些区域稀疏、哪些区域密集。

#### `cognitive hypotheses` — 潜在新连接

```
stg --agent stg-steam cognitive hypotheses
```

引擎根据当前图结构推测可能存在但还没记录的关系。结果是建议，需要你确认才会写入。

#### `cognitive route <查询>` — 多策略检索

```
stg --agent stg-steam cognitive route "Elden Ring features"
```

不只是 propagate —— 引擎尝试多种检索策略（语义搜索、社区跳转、路径查询），返回综合结果。

### 1.4 时间维度

#### `temporal range <日期>` — 某一天的活动

```
stg --agent stg-steam temporal range 2026-05-10
```

显示这一天新增了哪些边、修改了哪些节点。

#### `temporal around <节点名>` — 节点的时间邻域

```
stg --agent stg-steam temporal around Elden_Ring 24
```

显示这个节点出现前后 24 小时内还做了什么。

#### `temporal build <日期 [日期2]>` — 重建时间序列

```
stg --agent stg-steam temporal build 2026-05-10
stg --agent stg-steam temporal build 2026-05-08 2026-05-12
```

把这段时间内的边按 `created_at` 排序，建立"事件链"（一个连续的时间叙事）。

#### `temporal replay <节点>` — 时间回放

```
stg --agent stg-steam temporal replay Bug_G8_Dedup
```

从某节点开始，按时间顺序重走它周围所有边的事件链 —— 像回放一段记忆。

### 1.5 记忆整理（高级，日常勿用）

#### `merge` / `consolidate` / `xref`

```
stg --agent stg-steam merge '[A] -> [B] ::mod(confidence=0.95)'        # 补全已有边
stg --agent stg-steam consolidate --all --dry-run                       # 合并多边（先 dry-run 看）
stg --agent stg-steam xref --dry-run                                    # 扫 description 建跨社区虚拟边
```

这些是**显式触发**的记忆整理工具，不应在日常流程里。误用会破坏图结构。

### 1.6 Skill 调用

#### `use <skill 名>` — 执行已登记的 Skill

```
stg use My_Tool input_arg --option val
stg use My_Tool input_arg --dry-run                    # 看会执行什么但不真跑
stg use My_Tool input_arg --timeout 120                # 临时改超时
stg use My_Tool --args-stl '[X] -> [Y] ::mod(...)'    # STL 输入模式
```

需要先在配置里 opt-in：

```
stg config set skill.enabled true
stg config set skill.roots /abs/path/to/tools[,/path2]
stg config set skill.interpreters.myvenv /abs/path/to/bin/python3
```

#### `skill list` / `skill show` / `skill history` / `skill configure`

```
stg skill list                          # 所有可执行 Skill 节点
stg skill list --filter audit           # 按关键字过滤
stg skill list --all                    # 含不可执行的（仅文档登记）
stg skill show My_Tool                  # 详情 + 最近调用
stg skill history --limit 10            # 全部调用审计
stg skill configure My_Tool --executable --interpreter python3 --args-template "<in> [--out PATH]" --timeout 60
```

#### `propagate skill` — Skill 目录视图

```
stg propagate skill
```

特殊查询，渲染所有 Skill 节点的目录而不是普通社区视图。

### 1.7 工具杂项

| 命令 | 用途 |
|---|---|
| `stg search <查询>` | 语义检索（如果挂了 embedding） |
| `stg perceive <网格 JSON>` | 感知一帧画面，输出哈希 + 相似历史帧 |
| `stg perception stats` | 感知系统状态 |
| `stg telemetry status` | 遥测数据概览 |
| `stg telemetry frequency` | 节点真实激活频次排名 |
| `stg telemetry report` | 综合报告 |
| `stg simulate run --calibrated` | 用真实遥测数据驱动模拟 |
| `stg visualize` | 3D 星图可视化（4 种模式） |
| `stg tensions active` | 列出未解决的认知矛盾 |
| `stg alias add/list/remove/resolve` | 节点别名管理（同一概念多种写法归一化） |
| `stg guide` | 打印完整的 agent 操作手册（847 行） |
| `stg --help` | 全部命令列表 |

---

## 二、HTTP API（v1）

启动服务：

```
pip install stg-engine[server]
stg-server --agent stg-steam --port 8765
```

服务启动后：
- 接口根：`http://127.0.0.1:8765/v1/`
- 交互式 API 文档：`http://127.0.0.1:8765/docs`（Swagger UI）

### 2.1 服务端配置

```
stg-server --agent <名字>                       # 必填
           [--port 8765]                        # 端口
           [--bind 127.0.0.1]                   # 绑定地址
           [--log-level info]                   # 日志级别
           [--max-concurrent-propagate 4]       # 并发上限（M4 起生效）
           [--allow-external-bind]              # 非 localhost 绑定要求显式启用
           [--path <stg 文件路径>]              # 直接指定 .stg 文件（绕过 agent 解析）
```

### 2.2 端点详解

#### `GET /v1/health` — 存活探针

用途：编排器探活、客户端探测引擎是否还在、检测数据陈旧。

```bash
curl http://127.0.0.1:8765/v1/health
```

响应：
```json
{
  "status": "ok",
  "agent": "stg-steam",
  "node_count": 527,
  "edge_count": 8058,
  "server_version": "0.6.0a1",
  "engine_mtime": 1778565645.673902,
  "uptime_seconds": 5.69
}
```

`engine_mtime` 是底层 `.stg` 文件加载时的修改时间 —— 如果外部 ingest 改了文件，server 还停在旧版本，客户端可以靠这个字段检测。

#### `GET /v1/stats` — 完整图统计

```bash
curl http://127.0.0.1:8765/v1/stats
```

响应：

```json
{
  "agent": "stg-steam",
  "stats": {
    "node_count": 527,
    "edge_count": 8058,
    "real_edge_count": 3551,
    "virtual_edge_count": 4507,
    "psi": 100.0,
    "graph_density": 0.0217,
    "session_count": 0,
    "event_count": 0,
    "active_tensions": 0,
    ...
  }
}
```

#### `POST /v1/propagate` — 激活传播

最核心的查询端点。

请求体：

```json
{
  "query": "Elden Ring",              // 必填，1-2000 字符
  "max_nodes": 20,                    // 选填，返回的节点上限（默认 20，最大 200）
  "include_edges": true,              // 选填，是否返回每节点的出边（默认 true）
  "edge_limit_per_node": 10,          // 选填，每节点最多附带几条出边（默认 10）
  "full": false                       // 选填，是否包含 provenance 字段（默认 false）
}
```

示例：

```bash
curl -X POST http://127.0.0.1:8765/v1/propagate \
  -H "Content-Type: application/json" \
  -d '{"query": "Elden Ring", "max_nodes": 5, "edge_limit_per_node": 3}'
```

响应：

```json
{
  "agent": "stg-steam",
  "query": "Elden Ring",
  "elapsed_ms": 4,
  "seed_count": 1,
  "activated_count": 1,
  "nodes": [
    {
      "name": "Elden_Ring",
      "namespace": "Game",
      "activation": 0.21,
      "metadata_count": 32,
      "outgoing": [
        {
          "source": "Elden_Ring",
          "target": "FromSoftware_Inc",
          "confidence": 1.0,
          "strength": 0.5,
          "rule": null,
          "modifiers": {"action": "developed_by", "source": "steam_appdetails"}
        }
      ]
    }
  ],
  "truncated": false
}
```

注意：

- **modifiers 全是字符串**，即使原始数据有数字。客户端自己 cast。
- HTTP 路径下 propagate 自动以 `read_only=True` 跑 —— 不会触发 Hebbian、不写遥测、不改 active_context。

#### `GET /v1/node/{name}` — 节点详情

```bash
curl 'http://127.0.0.1:8765/v1/node/FromSoftware_Inc?edge_limit=10&full=false'
```

参数：
- 路径里的 `name` —— 节点名
- `?full=true` —— 包含 provenance 字段（source、created_at 等）
- `?edge_limit=N` —— 每方向最多 N 条边（默认 50）

响应：

```json
{
  "agent": "stg-steam",
  "node": {
    "name": "FromSoftware_Inc",
    "namespace": "Company",
    "anchor_type": null,
    "activation": 0.019,
    "tension": 0.0,
    "self_relevance": 0.0,
    "metadata": {},
    "metadata_count": 0,
    "incoming": [
      {
        "source": "Elden_Ring",
        "target": "FromSoftware_Inc",
        "confidence": 1.0,
        "strength": 0.5,
        "rule": null,
        "modifiers": {"action": "developed_by", "source": "steam_appdetails"}
      },
      {
        "source": "Elden_Ring",
        "target": "FromSoftware_Inc",
        "confidence": 1.0,
        "strength": 0.5,
        "rule": null,
        "modifiers": {"action": "published_by", "source": "steam_appdetails"}
      }
    ],
    "outgoing": []
  }
}
```

节点不存在 → HTTP 404。

#### `GET /v1/query` — 模糊搜索

```bash
curl 'http://127.0.0.1:8765/v1/query?pattern=elden&namespace=Game&limit=10'
```

参数：
- `pattern` —— 子串匹配，**可空**（空 pattern + namespace 列举该命名空间所有节点）
- `namespace` —— 可选，限定命名空间
- `limit` —— 上限（默认 50，最大 500）

响应：

```json
{
  "agent": "stg-steam",
  "pattern": "elden",
  "namespace_filter": "Game",
  "matches": [
    {
      "name": "Elden_Ring",
      "namespace": "Game",
      "edge_count_out": 50,
      "edge_count_in": 11
    }
  ],
  "total_matched": 1,
  "truncated": false
}
```

### 2.3 错误处理

| HTTP 状态 | 含义 | 客户端做法 |
|---|---|---|
| 200 | 正常 | 处理 body |
| 404 | 节点不存在 | 检查 name 拼写 |
| 422 | 请求字段不合格 | 看 detail，修改字段 |
| 500 | 引擎内部错误 | 检查服务日志，可能是数据问题 |
| 503 | 引擎还没准备好 | 短暂等待重试 |

错误响应统一形态：

```json
{
  "detail": "Node 'X' not in agent 'Y'."
}
```

（v1 没有 machine 路由码字段 —— v2 会加 `error` 字段）

### 2.4 待发布端点（M4）

下面这些会在 M4 milestone 加入：

| 端点 | 用途 |
|---|---|
| `GET /v1/attrs/{name}` | 节点 metadata 投影（包含 `?key=` 单 key 查询） |
| `GET /v1/paths?src=X&tgt=Y` | 两节点之间的语义路径 |

并发限流（429 Too Many Requests）也在 M4 加入。

### 2.5 安全模型（v1）

- 默认绑定 `127.0.0.1`，不暴露到外网
- 非 localhost 绑定**拒绝启动**，要求 `--allow-external-bind` 显式确认
- v1 没有 API key / 任何认证 —— 信任本机
- CORS：localhost 绑定时**完全开放**（方便浏览器原型）；非 localhost 绑定时**完全关闭**（强制使用反代终结）

如果要暴露到生产环境，等 v2 的认证层，或者用 nginx / Caddy 套一层反向代理 + auth。

---

## 三、Python 库

适用于把 STG 嵌进自己的 Python 程序。

### 3.1 安装与导入

```bash
pip install stg-engine
```

```python
from stg_engine import STGEngine
```

### 3.2 创建 / 加载引擎

```python
# 内存里新建一个空引擎
engine = STGEngine()

# 加载已有 .stg 文件
engine = STGEngine.load("/abs/path/to/memory.stg")

# 保存
engine.save("/abs/path/to/memory.stg")
```

注意：`STGEngine.load(path)` 是类方法，不是 `STGEngine()` + `engine.load(path)`。

### 3.3 录入

```python
# 单条 STL
engine.ingest_stl('[A] -> [B] ::mod(action="related_to", confidence=0.9)')

# 多条
stl_text = """
[A] -> [B] ::mod(confidence=0.9)
[B] -> [C] ::mod(confidence=0.8)
"""
count = engine.ingest_stl(stl_text)  # 返回成功 ingest 的边数

# 程序化 add_edge（绕过 STL 解析）
engine.add_edge("A", "B", confidence=0.9, action="related_to", source="manual")
```

### 3.4 查询

```python
# 节点
node = engine.get_node("A")
print(node.name, node.namespace, node.activation)

# 边
edges = engine.get_edges(source="A")          # A 的所有出边
edges = engine.get_edges(target="B")          # B 的所有入边
edges = engine.get_edges(source="A", target="B")

# 模糊搜索
matches = engine.query_nodes(name_pattern="cyber", namespace="Game", limit=20)

# 激活传播
activated = engine.propagate("Elden Ring", read_only=True)   # 返回节点名列表
metrics = engine._last_propagation_metrics                    # 看本次细节
```

### 3.5 学习与遥测

```python
engine.enable_learning()           # 开启 Hebbian
engine.disable_learning()
engine.enable_telemetry()          # 开启遥测
engine.disable_telemetry()

# 检查
if engine.learning_enabled:
    ...
if engine.telemetry_enabled:
    ...
```

### 3.6 嵌入式 HTTP server

```python
from stg_engine import STGEngine
from stg_engine.server import create_app, ServerState
from fastapi.testclient import TestClient

engine = STGEngine.load("/path/to/memory.stg")
state = ServerState(
    engine=engine,
    agent_name="my-agent",
    stg_path="/path/to/memory.stg",
    engine_mtime=...,
    server_version="0.6.0a1",
)
app = create_app(state, allow_cors=True)
client = TestClient(app)
response = client.post("/v1/propagate", json={"query": "X"})
```

可以挂到自己的 FastAPI app、Starlette app、或嵌到测试里。

### 3.7 关键类型

- `STGEngine` —— 引擎主对象
- `STGNode` —— 节点（字段：name, namespace, anchor_type, activation, tension, self_relevance, metadata）
- `STGEdge` —— 边（字段：source, target, confidence, strength, rule, salience, created_at, last_used, modifiers, ...）
- `STGSession` / `STGEvent` / `STGTension` —— 会话相关元数据
- `PropagationMetrics` —— `_last_propagation_metrics` 的类型

---

## 四、配置文件

位置：`~/.stg/config.json`

通过 `stg config get/set/list` 操作：

```bash
stg config list                                  # 列全部
stg config get skill.enabled
stg config set skill.enabled true
stg config set skill.roots "/abs/path/tools[,...]"
stg config set skill.interpreters.python3 "/abs/path/bin/python3"
```

常用键：

| 键 | 值 | 用途 |
|---|---|---|
| `skill.enabled` | true/false | Skill 执行总开关 |
| `skill.roots` | 路径列表 | Skill 脚本白名单目录 |
| `skill.interpreters.<name>` | 解释器绝对路径 | 命名解释器映射（脚本可用 `interpreter=<name>`） |

---

## 五、目录结构

### Agent 数据

```
~/.stg/
├── config.json
├── last_ingest.json          # 最近一次 ingest 状态
├── last_propagate.json       # 最近一次 propagate 状态
└── <agent-name>/
    └── memory.stg            # 主存储文件（SQLite + 二进制）
```

每个 agent 一个目录。切换 agent：所有命令前加 `--agent <name>`。

### 工具数据（Skill）

由 `skill.roots` 配置决定，没有约定路径。

---

## 六、典型完整流程

### 流程 A：从零建一个游戏知识库

```bash
# 1. 准备数据文件 elden.md（每行一条 STL）
cat > /tmp/elden.md <<'EOF'
[Elden_Ring] -> [FromSoftware_Inc] ::mod(action="developed_by", confidence=1.0)
[Elden_Ring] -> [FromSoftware_Inc] ::mod(action="published_by", confidence=1.0)
[Elden_Ring] -> [Soulslike] ::mod(action="belongs_to_genre", confidence=0.95)
[FromSoftware_Inc] -> [Japan] ::mod(action="based_in", confidence=1.0)
EOF

# 2. ingest
stg --agent games ingest-file /tmp/elden.md

# 3. 检查
stg --agent games stats
stg --agent games node Elden_Ring

# 4. 查询
stg --agent games propagate "Elden Ring"

# 5. 暴露给 AI agent
stg-server --agent games --port 8765
# 客户端：curl -X POST http://127.0.0.1:8765/v1/propagate ...
```

### 流程 B：日常使用某个 agent

```bash
# 早上开始
stg --agent syn-claude stats                          # 看看昨天的状态
stg --agent syn-claude tensions active                # 有未解决的认知矛盾吗

# 工作中
stg --agent syn-claude propagate "STG HTTP server"     # 想了解什么
stg --agent syn-claude node Bug_G8_Dedup_Issue         # 看具体节点

# 学到了新东西
stg --agent syn-claude ingest '[New_Concept] -> [Existing_Hub] ::mod(action="extends", confidence=0.9, description="...")'
stg --agent syn-claude bind 1,2                        # 绑到候选社区

# 工作结束
stg --agent syn-claude feedback session-end            # 保存 + 整理 + 刷新
```

### 流程 C：debug 某个 bug 的历史

```bash
stg --agent syn-claude temporal range 2026-05-12       # 那天发生了什么
stg --agent syn-claude temporal around Bug_G8_Dedup 24 # 这个节点出现前后做了什么
stg --agent syn-claude temporal build 2026-05-12       # 把那天串成事件链
stg --agent syn-claude temporal replay Bug_G8_Dedup    # 沿事件链回放
```

---

## 七、Cheatsheet（一页速查）

```
# 查
stg stats
stg node <名字>
stg query <pattern>
stg attrs <名字> [--keys | --key <name>]
stg propagate <文本>

# 写
stg ingest '<STL>'
stg ingest-file <路径>
stg bind <序号>
stg select <序号>

# 学习
stg learn path <n1> <n2> ...
stg feedback session-end

# 时间
stg temporal range <日期>
stg temporal around <节点> [小时]
stg temporal build <日期>
stg temporal replay <节点>

# Skill
stg use <skill 名> [args]
stg skill list / show / history / configure

# HTTP
pip install stg-engine[server]
stg-server --agent <名字>
GET  /v1/health   /v1/stats
POST /v1/propagate
GET  /v1/node/{name}
GET  /v1/query

# Agent 切换
stg --agent <name> <command>
```

---

**版本**：本手册对应 stg-engine 0.6.0a1（HTTP server M1-M3 已上）。
**更新于**：2026-05-12。
**配套**：架构总览见 `SPECIFICATION.md`。
