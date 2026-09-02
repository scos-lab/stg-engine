# STG Engine — 规格说明书

> 这是一份**宏观说明**。讲 STG Engine 是什么、由哪些部分组成、各部分做什么、怎么配合工作。
> 不讲数学公式、算法细节、源代码结构。如需深入，参考 SKC 仓库中的 design docs。
> 配套 API 操作手册：`API_REFERENCE.md`。

---

## 1. STG Engine 是什么

STG Engine 是一个**语义张力图引擎** —— 把知识表示为一张带有方向和强度的图，并在这张图上进行存储、推理、学习。

它面向三类问题：

- **AI agent 的长期记忆**：让一个 LLM agent 跨会话保留知识、形成偏好、自然遗忘
- **结构化知识库**：把领域知识（游戏、书籍、人物、事件）组织成可查询的图，供 Q&A / RAG 消费
- **可计算的语义网络**：知识不只是被存起来，还能"激活传播" —— 提到一个概念，相关的概念自动浮现

与向量数据库相比：STG 保留**结构**（A 因为 B 而导致 C），不是把一切压成向量再算相似度。

与传统知识图谱数据库相比：STG 自带**激活传播 / 学习 / 遗忘**等动力学机制，不只是静态查询。

---

## 2. 核心概念

只需理解六个概念，就能用 STG。

### 节点（Node）

一个**具名实体**。例如 `Elden_Ring`（一款游戏）、`FromSoftware_Inc`（一家公司）、`Lesson_OAuth_Expiry`（一条经验教训）。

每个节点有：
- **名字**（唯一标识，PascalCase 或下划线分隔；支持中文）
- **命名空间**（可选，例如 `Game`、`Company`、`Bug`）—— 把节点分到不同的"类别"里
- **属性**（intrinsic properties，例如 Elden_Ring 的发行年份、价格） —— 像一个小型的 key-value 卡片

### 边（Edge）

一个**有方向的语义关系**。`[Elden_Ring] → [FromSoftware_Inc]` 表示 Elden_Ring 与 FromSoftware_Inc 之间有联系，方向从前者指向后者。

每条边有：
- **来源** / **目标**（两个节点）
- **修饰符**（modifiers）—— 一组键值对，描述这条关系的性质（见下）
- **可信度（confidence）**：0-1 之间，这条声明有多可信
- **显著性（salience）**：0-1 之间，这条边眼下有多"亮"（会随时间衰减，会被使用强化）

### 修饰符（Modifier）

边上的标签，描述这是哪种关系。最重要的几种：

- **语义场（semantic field）**：`action` / `is_a` / `role` / `status` / `relation` / 其中之一 —— 必填一个，标明这是动作、归类、角色还是状态。例：`action="developed_by"` 表示"开发于"。
- **置信度** `confidence` —— 必填
- **规则类型** `rule`：`causal`（因果）/ `logical`（逻辑）/ `empirical`（经验）/ `definitional`（定义）
- **来源** `source` —— 引用、URL、文献号等
- **发生时间** `occurred_time` —— 事件发生的真实时间（区别于"被记录的时间"）
- **描述** `description` —— 一句话补充
- **路径** `path` —— 如果这条边引用一个文件，写文件路径

### 命名空间（Namespace）

节点的分类标签。例如 `Game:Elden_Ring`、`Company:FromSoftware_Inc`、`Bug:G8_Dedup_Issue`。

命名空间不是节点的"父级"，只是"它属于哪一类"。同一个名字在不同命名空间里**不能**有两个节点 —— 命名空间是分类标签，不是层级。

### Agent（智能体 / 知识库实例）

一份独立的 STG 数据。一个 agent 对应一个**目录**（默认在 `~/.stg/<agent_name>/`），里面有这个 agent 的所有节点、边、会话记录、学习日志等。

你可以同时有多个 agent：`syn-claude`、`stg-steam`、`bible-nwt`、`linux-source-stl`...... 每个互不干扰，各自演化。

### STL（Semantic Tension Language）

写入 STG 的**文本语法**。形式：

```
[源节点] -> [目标节点] ::mod(键=值, 键=值, ...)
```

例：

```
[Elden_Ring] -> [FromSoftware_Inc] ::mod(action="developed_by", confidence=1.0, source="steam_appdetails")
```

STL 是一种小型领域语言（DSL），由 stl-parser 库解析。LLM 容易生成，人类容易写。

---

## 3. 系统由哪些部分组成

STG Engine 可以分成若干**功能模块**。它们不是各自独立的"产品"，而是同一个引擎的不同侧面，互相协作。

### 3.1 存储与持久化

把内存里的图保存到磁盘，再读回来。

- **存储格式**：`.stg` 文件（SQLite 数据库 + 内嵌的二进制结构）
- **目录约定**：每个 agent 一个目录，里面有主图、日志、配置
- **加载**：启动时一次性读入内存，之后所有查询都在内存里跑

### 3.2 知识录入（Ingest）

把外部信息写进 STG。三个层次：

1. **直接 STL 语句**：手写一条边，最常用
2. **批量文件**：把多条 STL 写进一个 `.md` 或 `.stl` 文件，一次性导入
3. **外部源导入**：从 Obsidian 笔记、Markdown 文档、记忆矩阵、通用文本来源批量摄取（importers 模块负责）

录入时引擎会自动做几件事：
- **解析与校验**：语法错、字段不全 → 提示修复
- **去重**：相同的边不重复存（"相同"的判定基于来源、目标、语义场、置信度等等）
- **冲突检测**：与已有知识矛盾时打 warning（不会拒绝写入）
- **候选社区提示**：新节点告诉你"可以绑到哪几个已有社区"

### 3.3 知识检索（Query / Propagate）

从 STG 里取出信息。三种典型查询：

1. **精确查询**：节点详情、边详情、按命名空间筛选
2. **模糊搜索**：节点名包含某个字符串
3. **激活传播（propagate）**：给一段文本，引擎从匹配到的节点出发，沿边激活相关节点 —— 这是 STG 最核心的"语义检索"

激活传播的语义：
- 不只是看一跳邻居，而是顺着边的强度向外扩散
- 高 salience 的边传得远，低 salience 的边传得近
- 多种模式可选：粗粒度（看大方向）/ 细粒度（看细节）/ 双锚点（精确实体对查询）/ 自动

### 3.4 学习与遗忘

STG 不只是被动的存储，它会**根据使用情况调整自己**。

- **Hebbian 学习**：被激活的边变得更显著（salience 上升），长期没被激活的边逐渐褪色
- **时间衰减**：所有边的 salience 都会随时间慢慢衰减（半衰期 30 天）
- **重要性**：高频被访问的节点会被标记为"重要"，传播时优先考虑

这套机制是可关可开的。HTTP 服务路径下默认**关闭**（外部访问不应该污染 agent 自己的学习信号），CLI 路径下默认**开启**（你自己用引擎，引擎记住你关心什么）。

### 3.5 时间维度

STG 把时间当成头等公民。每条边都有两个时间字段：

- **occurred_time**：事件**发生**的真实时间（你写的）
- **created_at**：这条边**被记录**的时间（引擎自动设的，或 ingest 时显式覆盖）

基于此可以做：
- **时间范围查询**：某天 / 某月新增了什么知识
- **时间序列重建**：把一段时间内的边按发生顺序串起来，回放当时的"思维轨迹"
- **回溯调试**：某个 bug / 决定的前后上下文

### 3.6 重力与社区（Gravity）

如果把激活传播比作"水流"，那"重力图"就是地形。

- 引擎自动识别**节点社区**（彼此高密度连接的节点群） —— 比如 stg-steam agent 里 "Soulslike 游戏群"、"Steam 客户端相关节点群"
- 每个社区有**代表节点**（hub），命名空间用"主题词"自动命名
- 每个节点有**海拔（elevation）**：在社区里越中心，海拔越高
- 激活传播默认开启重力 —— 沿地形流，更准确

### 3.7 矛盾、合并、整理（Merge / Consolidate / Supersede）

记忆需要打扫。

- **Supersede 检测**：同一个事实在不同时间有不同陈述时，引擎给旧版本打标记（不删除，保留历史）
- **Merge（合并）**：把零散的边补全成一条更完整的边
- **Consolidate（整合）**：把多条同主题的边合并为一条
- **XRef（交叉引用）**：扫描 description 里提到的节点名，自动建立跨社区虚拟边

这套整理工具**不应在日常使用** —— 它是"打扫记忆"的工具，由用户显式触发。

### 3.8 感知（Perception，可选）

把视觉网格（例如 ARC-AGI 任务里的小色块图）转成 STG 节点。

- 用一组卷积滤波器把像素映射为特征
- 相似的画面映射到同一/相邻节点
- 用于把"看到什么"接入 STG 的"知道什么"

只在视觉 / 游戏 / 感知-动作循环场景下需要。普通 RAG / 知识库用法用不上。

### 3.9 可执行 Skill

STG 节点可以不只是数据，也可以是**可执行命令**。

- 在 STG 里登记一个 `Skill:XXX` 节点，附上脚本路径、解释器、参数模板、超时
- 之后 `stg use XXX` 直接调起，subprocess 跑、有审计、有超时保护
- 安全门：用户先在配置里 opt-in，再给具体 edge 标 `executable="true"`，脚本必须在白名单目录下

这把 STG 从"记忆"扩展成"记忆 + 能力"。Agent 可以记住"这件事用哪个工具做"并直接调用。

### 3.10 HTTP 服务（最近新加）

开一个 HTTP 服务进程，把 STG 的**查询能力**通过 JSON API 暴露出来。

- 一个 agent 对应一个进程（一个端口）
- 五个核心端点：健康、统计、激活传播、节点详情、模糊查询
- 默认绑定 localhost，外部访问需要显式开启
- HTTP 路径下 propagate 自动以 `read_only=True` 运行（外部流量不污染 agent 学习）

为什么有它？因为子进程调用 CLI 每次都要付 ~50ms 的 Python 启动开销。AI agent 高频 Q&A 时不可接受。HTTP 服务把一次加载摊销给后续所有请求。

### 3.11 工具与可视化

附带几个调试 / 探索工具：

- **stats** —— 整体规模、密度等
- **dump** —— 分页浏览整张图
- **propagate -v** —— 看激活流向哪里
- **visualize** —— 3D 星图渲染（4 种模式）
- **telemetry** —— 真实使用数据采集（频次、延迟等），用于事后分析

---

## 4. 典型工作流

### 4.1 建立一个新知识库

```
1. 选名字（例：`my-kb`）
2. 写一个 .md 文件，放入若干条 STL 语句
3. stg --agent my-kb ingest-file path/to/notes.md
   → 自动创建 ~/.stg/my-kb/ 目录，新建 .stg 文件，写入边
4. stg --agent my-kb stats
   → 确认节点 / 边数量符合预期
```

### 4.2 日常查询

```
1. stg --agent my-kb propagate "Elden Ring system requirements"
   → 引擎激活相关节点，按相关度排序，按社区聚合显示
2. stg --agent my-kb node Elden_Ring
   → 看这个节点的全部入边 / 出边 / 元数据
3. stg --agent my-kb query "elden"
   → 按名字模糊搜
```

### 4.3 知识更新

```
1. stg --agent my-kb ingest '[Elden_Ring] -> [DLC:Shadow_of_the_Erdtree] ::mod(action="has_dlc", confidence=1.0, occurred_time="2024-06-21")'
   → 引擎吃下新边，更新内存图
2. stg --agent my-kb feedback session-end
   → 保存到磁盘 + 刷新遥测
```

### 4.4 暴露给 AI agent / 其他客户端

```
1. pip install stg-engine[server]
2. stg-server --agent my-kb --port 8765
   → 启动 HTTP 服务
3. 客户端（LLM / 浏览器 / 其他语言）POST /v1/propagate, GET /v1/node/X 等
```

### 4.5 让 STG 学习

```
1. 平时正常用 propagate / select 等命令，Hebbian 自动开启
2. session 结束：stg --agent my-kb feedback session-end
   → 修剪低 salience 的边、保存遥测
3. 偶尔 stg --agent my-kb cognitive self-model
   → 看 agent 自己评估"哪些知识区域薄弱 / 强项在哪"
```

---

## 5. 多 Agent 模型

你可以拥有任意多个 agent，每个完全独立：

| Agent | 用途 |
|---|---|
| `syn-claude` | 我（Syn-claude）的长期记忆 |
| `stg-steam` | Steam 游戏知识库 |
| `bible-nwt` | 圣经新世界译本研究 |
| `linux-source-stl` | Linux 内核源码知识图 |

切换 agent：`stg --agent <name> <command>` 或 `stg-server --agent <name>`。

agent 之间默认互不可见。一个 agent 学到的东西不会跑到另一个里。这是设计选择 —— 跨 agent 的知识桥接是更高一层的协调问题，不归 stg-engine 负责。

---

## 6. 数据流：从录入到检索的全程

为了帮你建立心理模型，下面是一条 STL 语句从写入到再被读出的完整旅程：

```
[1] 你手写或 LLM 生成 STL 文本
        │
        ▼
[2] stl-parser 解析为结构化对象
        │
        ▼
[3] 引擎做语义校验（必填字段 / 字段值范围）
        │
        ▼
[4] 去重检查（同源 + 同目标 + 同语义场 + 同其他字段 → 视为重复）
        │
        ▼
[5] 冲突检测（与已有边是否矛盾，仅 warning）
        │
        ▼
[6] 写入内存图：节点池、边池、邻接索引
        │
        ▼
[7] 候选社区提示（新节点告诉你"可以挂在哪些已有 hub 上"）
        │
        ▼
[8] save：内存图序列化到 ~/.stg/<agent>/memory.stg
        │
        ▼
        ... 之后某个时刻 ...
        │
        ▼
[9] propagate("查询文本") 启动
        │
        ▼
[10] 文本分词 + 匹配节点 → 找到"种子节点"
        │
        ▼
[11] 从种子节点向外激活传播：沿边走，沿重力图走
        │
        ▼
[12] 收集激活到阈值以上的节点，按社区聚合，按激活值排序
        │
        ▼
[13] CLI/HTTP 渲染结果（含 community 标签、激活值、出边）
        │
        ▼
[14] 如果 read_only=False：Hebbian 学习更新 salience，遥测记录
```

---

## 7. 我能用 STG 做什么 / 不能做什么

**适合做的**：

- AI agent 的长期、可演化、可回顾的记忆
- 中等规模（几万 ~ 几十万节点）领域知识库的检索后端
- 知识演化追踪：什么时候记录了什么、为什么改了主意
- 多 hop 推理（"A 影响 B 影响 C，那 A 和 C 有什么关系？"）
- 跨文档桥接（不同来源谈到同一概念，自动连起来）

**不适合做的**：

- 海量（千万级以上）数据 —— 不是 OLTP 数据库
- 严格的事务性需求（金融、订单）—— STG 不保证 ACID
- 实时低延迟查询（个位毫秒级）—— 激活传播是 50-500ms 量级
- 完全替代向量数据库 —— STG 和 embedding 检索是互补的，不是替代

---

## 8. 限制与注意事项

- **图必须 fit 在内存里**：当前架构不分片。100K 节点级别还很轻松，1M 节点级别要审视
- **写入路径单线程**：HTTP 服务只暴露读，不暴露写，正是为了避免并发写入问题
- **学习是默认开启的**：CLI 用户要意识到自己的 propagate 会改变 agent 的 salience 分布（这是 feature 不是 bug，但要知道）
- **`.stg` 文件不要并发写**：一个 agent 同时只应该有一个写者（一个 CLI 进程 或 一个 server，但 server 在 v1 还不写）
- **agent 切换会重建缓存**：跨 agent 频繁切换会重复加载，单一 agent 长会话更高效

---

## 9. 与相邻技术的关系

| 系统 | 它擅长什么 | STG 擅长什么 | 关系 |
|---|---|---|---|
| 向量数据库（Pinecone, Chroma, Qdrant） | 高维相似度检索 | 结构化语义关系 | **互补** —— STG 节点可以挂 embedding，但 STG 不靠 embedding 推理 |
| 图数据库（Neo4j, ArangoDB） | 通用图查询，事务 | 激活传播 + 学习 + 衰减 | **不同侧重** —— STG 更像"活的"图，传统图 DB 更像"静的"图 |
| 知识图谱（RDF, OWL） | 严格本体论、推理引擎 | 模糊语义、自演化 | **不同哲学** —— STG 接受模糊 / 矛盾 / 演化，RDF 追求严格一致 |
| 经典 RAG（向量 + LLM） | 文本片段检索 | 结构 + 关系 + 时间 | **可叠加** —— STG 可作为 RAG 的二级精炼层 |

---

## 10. 怎么开始

阶段一（5 分钟）：

1. `pip install stg-engine`
2. 跟着 README 的 Quickstart 跑通"两条边 + 一次 propagate"
3. 看到激活传播在屏幕上展开

阶段二（30 分钟）：

1. 选一个你熟悉的小领域（一本书、一个游戏、一个项目）
2. 写 50 条 STL 边，用 `stg ingest` 批量录入
3. 试着用不同的 propagate 查询，观察激活路径

阶段三（按需）：

1. 看 `STG_AGENT_GUIDE.md` —— AI agent 操作员手册
2. 看 SKC 仓库的 design docs —— 想理解为什么这样设计时
3. 看本仓库 `API_REFERENCE.md` —— 查具体命令和端点的用法

---

**版本**：本说明书对应 stg-engine 0.6.0a1（HTTP server v1 已上 M3）。
**更新于**：2026-05-12。
