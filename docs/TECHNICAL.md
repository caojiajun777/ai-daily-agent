# AI 日报 — 技术文档

## 项目概述

全自动 AI 科技日报生成系统。每日从 46 个全球化信息源采集 AI 领域最新资讯，经 10 个角色化 Agent 协作流水线处理后，生成结构化的中文深度日报，发布至 GitHub Issues 并推送 PDF 至订阅者邮箱。

**GitHub:** [github.com/caojiajun777/ai-daily-agent](https://github.com/caojiajun777/ai-daily-agent)

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    AI 日报自动化流水线                        │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Collector│→│ Curator  │→│  Writer  │→│  Critic  │   │
│  │ 46 sources│  │ Cluster  │  │deepseek- │  │ 幻觉检测 │   │
│  │ RSS/X/Web│  │ Score    │  │chat      │  │ 语义去重 │   │
│  │ arXiv    │  │ LLM Edit │  │          │  │          │   │
│  └──────────┘  └──────────┘  └──────────┘  └────┬─────┘   │
│                                                  │         │
│                           ┌──────────┐           │         │
│                           │ Repairer │←──────────┘         │
│                           │ 自动修复  │                     │
│                           └──────────┘                     │
│                               │                            │
│              ┌────────────────┴──────────────┐             │
│              ↓                               ↓             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │ Publisher│  │   PDF    │→│  Email   │               │
│  │ Issue #N │  │ Generator│  │ SMTP QQ  │               │
│  └──────────┘  └──────────┘  └──────────┘               │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │ Web 看板 │  │ 健康检查 │  │  告警    │               │
│  │ FastAPI  │  │ RSS 新鲜度│  │ 邮件通知 │               │
│  │ 归档/统计 │  │ 流水线状态│  │ 连续失败 │               │
│  └──────────┘  └──────────┘  └──────────┘               │
└─────────────────────────────────────────────────────────────┘
```

---

## 各模块详解

### 1. 数据采集层 (`agent/sources/`)

| 适配器 | 类型 | 说明 |
|--------|------|------|
| `rss.py` | RSS/Atom | 标准 RSS 2.0 和 Atom 1.0，feedparser 解析，HTML 标签清洗 |
| `arxiv_adapter.py` | arXiv API | 免费公开 API，按 cs.AI/CL/LG/CV 分类查询，top venue 过滤（NeurIPS/ICML/ICLR 等） |
| `x_cookie_adapter.py` | X Playwright | 浏览器渲染抓取 `x.com/{username}`，CI 环境自动直连（无需代理），提取真实推文 URL 和时间戳 |
| `x_adapter.py` | X API v2 | Bearer Token 认证，按时间窗口过滤推文（需付费 API） |
| `aihot_adapter.py` | Web Scraper | AI HOT Daily 中文聚合源，提取结构化条目及来源标注 |
| `base.py` | 工厂模式 | 统一 `SourceAdapter` 协议 + `build_source()` YAML 驱动工厂 |

**关键指标：**
- 46 个有效信源（23 X/Twitter + 20 RSS + arXiv + AI Hot + 36kr）
- 信源覆盖：OpenAI、Anthropic、Google DeepMind、Meta AI、百度、阿里云 Qwen、阶跃星辰、腾讯混元等
- 单次采集 200-270 条原始条目（X 源修复后）
- 语言：中文 + 英文双语覆盖
- X 源在 CI 环境（美国 runner）自动直连，无需代理

### 2. 内容策展层 (`agent/agents/`)

**v2.2 默认流程（research_editor 模式）：**

1. **EventClusterer** — 标题哈希 + URL 去重 + 标题相似度聚类（>0.68）
2. **EventScorer** — 多维度规则评分
3. **HistoryChecker** — 从已发布 Issue 加载近期标题，去重
4. **ResearchEditor (LLM)** — deepseek-chat 模型，基于候选事件做出编辑决策（select/reject、priority、section）
5. **FinalSelector** — section 多样性 + source 多样性 + 核心板块软覆盖 + 回退机制

**关键算法：**
- **时效性衰减**：指数衰减模型，72 小时半衰期
- **来源多样性惩罚**：同一源第 4 条 ×0.85，第 7 条 ×0.60，防止单源霸榜
- **AI 相关性保证**：arxiv/hf_daily_papers 自动视为 AI 相关（`_AI_GUARANTEED_SOURCES`）
- **技术洞察限量**：论文、研究、安全和 Benchmark 进入“技术与洞察”，不再强制 min 5 篇论文
- **回退机制**：LLM 选条不足时自动按 rule_score 填充差额，但不强制塞满每个板块

### 3. LLM 写稿层 (`agent/agents/writer.py`)

- **模型**：deepseek-chat（非推理模型，结构化 JSON 输出稳定）— **禁止使用 v4-pro**
- **输出格式**：7 个 Juya-style 内容板块（要闻/模型发布/开发生态/技术与洞察/产品应用/行业动态/前瞻与传闻）；Markdown 顶部概览只展示非空板块
- **每条条目**：150-350 字深度分析 + 2-4 条要点提炼
- **容错机制**：
  - Think block 自动剥离
  - JSON 修复引擎（尾逗号删除、缺失逗号补全、字符串拼接修复）
  - 日期覆盖保护（不信任 LLM 返回的日期字段）

### 4. 质量审核层 (`agent/agents/critic.py`)

**两层审核架构：**

| 层级 | 类型 | 检查项 |
|------|------|--------|
| 第一层 | 确定性审核 | URL 白名单校验、板块/条目数量检查、禁止短语检测、重复标题检测 |
| 第二层 | 语义去重 | LLM 调用检测跨板块同一事件重复（high/medium/low 三级严重度） |

**Publish Gate 规则：**
- 必须通过 critic + eval 检查
- 语义重复在 repair 成功后降级为 warning（不阻塞）
- 相同 URL 的语义重复降级为 warning（可自动合并）
- `--force-dup` 只跳过重复检查；`--force` 跳过全部门禁

### 5. 自动修复层 (`agent/agents/repairer.py`)

- 触发条件：语义去重检测到 high/medium 级别重复
- 修复操作：移除/替换重复条目，维持 section 不空
- 防注入：替换 URL 必须在策展白名单内

### 6. 配图提取层 (`agent/tools/image_extractor.py`)

**双层提取架构：**

| 层级 | 技术 | 适用场景 |
|------|------|---------|
| 第一层 | HTTP og:image | 传统服务端渲染网站 |
| 第二层 | Playwright headless | JS SPA 网站 |

跳过域名：ithome.com, x.com, twitter.com, github.com, youtube.com, arxiv.org

### 7. 发布层 (`agent/tools/issue_publisher.py` & `agent/agents/issue_publisher.py`)

- GitHub Issues API 发布（PAT 认证）
- 重复检测（30 天内标题 + 日期匹配）
- `--force-dup` 只跳过重复检测（不跳过质量门禁）
- `--force` 跳过质量门禁 + 重复检测
- `--dry-run` 本地预览

### 8. PDF + 邮件层 (`agent/tools/pdf_emailer.py`)

- MD → HTML → PDF：Playwright Chromium 渲染，A4 格式，中文字体
- SMTP 邮件：QQ 邮箱 SMTP SSL（465 端口），HTML 正文 + PDF 附件
- 订阅管理：Issue 评论自动订阅/退订（`subscribe.yml` workflow）

### 9. Web 看板 (`agent/web/`)

| 路由 | 功能 |
|------|------|
| `/` | 首页 — 最新日报全文 + 系统概览统计 |
| `/archive` | 归档 — 所有历史报告列表（日期/条目数/状态） |
| `/stats` | 统计 — 信源分布柱状图 + Token 用量 + 14 天历史 |
| `/monitor` | 监控 — 信源健康表 + 流水线成功率 + 成本追踪 |
| `/report/{date}` | 单篇报告查看 |

**REST API：**
- `GET /api/reports` / `GET /api/reports/{date}` — 结构化日报数据
- `GET /api/stats` / `GET /api/monitor` — 统计 + 健康检查
- `POST /api/subscribe?email=xxx` / `POST /api/unsubscribe?email=xxx`

### 10. 源发现系统 (`agent/agents/source_scout.py` + `agent/tools/auto_admit.py`)

**三通道发现：**
- LLM 语义发现（deepseek-chat）— 分析覆盖盲区，推荐新 RSS/X 源
- 内容链扩散 — 从已采集文章的域名中探测 RSS
- 社交图扩散 — X follow graph 遍历（需 X_BEARER_TOKEN）

**自动纳新流程：**
- `scout` → 输出 `scout_{date}.json` → `admit-sources` 去重 → 追加到 `default.yaml`（低权重 0.55）
- CI 当前做 dry-run 预览，不自动纳新（需人工 review）

### 11. 监控层 (`agent/tools/monitor.py`)

- **信源健康**：定期检查所有 RSS feed 新鲜度，按 ok/stale/dead 分级
- **流水线健康**：14 天历史成功率、Token 用量、成本统计
- **告警**：连续失败/信源失效自动邮件通知

### 12. 基础设施层 (`agent/harness/`)

| 组件 | 功能 |
|------|------|
| `state.py` | 流水线状态机（pending/running/ok/failed/needs_human_review/skipped） |
| `trace.py` | Append-only JSONL 日志，含 prompt hash、token 估算、LLM 调用追踪 |
| `budget.py` | Token 预算管理，分阶段归属，硬上限保护（200K in / 30K out / 40 calls） |
| `tools.py` | ToolRegistry + JSON-Schema 参数验证 |
| `context.py` | 滚动窗口消息裁剪 |

### 13. LLM 抽象层 (`agent/llm/`)

- `LLMProvider` Protocol：统一接口
- `DeepSeekProvider`：OpenAI-compatible SDK，支持 `deepseek-chat` 和 `deepseek-v4-pro`
- `MockLLMProvider`：离线测试用，自定义 responder 函数
- **关键规则**：需要结构化 JSON 的 Stage 用非推理模型，禁止推理模型
- **自定义 API 网关约束**：`DEEPSEEK_BASE_URL` 仅暴露 `deepseek-v4-flash` + `deepseek-v4-pro`，`deepseek-chat` 需 `skip_model_check=True` 才能使用（网关仍接受该模型名并正确路由）

---

## 技术栈

| 类别 | 技术 | 用途 |
|------|------|------|
| 语言 | Python 3.12+ | 全部后端逻辑 |
| LLM | DeepSeek chat + v4-pro | 日报写稿 + 编辑筛选 + 语义去重（chat），少数任务（v4-pro） |
| Web 框架 | FastAPI + Jinja2 | Web 看板 + REST API |
| 数据采集 | feedparser, httpx, Playwright, X API v2 | 多协议采集 |
| 浏览器自动化 | Playwright (Chromium/Edge) | X 页面抓取 + PDF 生成 |
| 数据建模 | Pydantic v2 | 全流水线数据验证 |
| 测试 | pytest (138+ tests) | 离线 mock 全覆盖 |
| CI/CD | GitHub Actions | 定时日报 + 订阅 + 健康检查 + 测试（4 workflows） |
| 容器化 | Docker Compose | 一键部署 |
| 监控 | 自研 monitor.py | 信源/流水线/成本追踪 |
| 配置 | YAML + 环境变量 | 分层配置管理 |

---

## 关键指标

### 系统规模

| 指标 | 数值 |
|------|------|
| 信源数量 | 46 (23 X + 20 RSS + arXiv + AI Hot + 36kr) |
| Agent 数量 | 10 (Collector/Clusterer/Scorer/History/Editor/FinalSelector/Writer/Critic/SemanticDup/Repairer/Publisher) |
| CI Workflows | 4 个 (日报/订阅/健康检查/测试) |

### 运行指标

| 指标 | 典型值 |
|------|--------|
| 单次采集量 | 200-270 条 |
| 聚类后事件数 | 80-115 |
| 策展后数量 | 14-18 条 |
| LLM 调用次数 | 3-5 次/天 (editor + writer + semantic dup + repair) |
| Token 消耗 | ~15,000 input + ~10,000 output |
| 单次成本 | ~$0.005 |
| 日报板块 | 7 个 Juya-style 内容板块；顶部概览只展示非空板块 |
| 信源语言 | 中文 + 英文 |
| 论文配额 | 无硬配额，通常 0-3 篇/day |

### 质量指标

| 指标 | 当前值 |
|------|--------|
| 标题唯一率 | 100% |
| 幻觉链接率 | 0%（URL 白名单强制校验） |
| 禁止短语命中率 | 0% |
| 语义重复率 | 0%（自动修复） |
| LLM 编辑选择率 | 24/82 → 最终 19 条（v2.2 已修复） |

---

## 简历可用表述

### 中文版

> **AI 日报全自动生成系统** | Python, FastAPI, DeepSeek LLM, GitHub Actions
>
> - 独立设计并实现了 10-Agent 协作流水线架构，覆盖多协议数据采集、事件聚类与评分、LLM 编辑决策、深度写稿、多层质量审核、语义去重与自动修复、发布推送全链路
> - 构建了 46 个全球化信息源聚合系统，支持 RSS、X/Twitter（Playwright 浏览器渲染 + API v2 双通道）、arXiv 学术 API、Web Scraper 多协议接入；含 AI 相关性过滤、来源多样性惩罚、论文保底配额等策展算法
> - 基于 FastAPI 构建 Web 看板及 REST API，含日报归档、信源分布统计、系统健康监控、成本追踪
> - 实现 X/Twitter 浏览器端抓取（真实链接 + 时间戳提取、CI 免代理直连）、Playwright headless 配图自动提取、PDF 生成及 SMTP 邮件推送
> - 全 GitHub Actions CI/CD 编排（定时日报 + 订阅管理 + 健康检查 + 自动化测试）；多环境兼容（中国代理 / CI 直连自适应）
> - 深度排查 LLM 输出截断/JSON 解析失败等问题，针对性优化 token 预算、模型选择、解析容错

### 英文版

> **AI Daily Report — End-to-End Automated News Generation System**
>
> - Architected a 10-Agent collaborative pipeline (Collect → Cluster → Score → LLM Edit → Write → Critique → Semantic Dedup → Repair → Publish) processing 46 global sources into structured Chinese daily reports
> - Built multi-protocol source aggregation: RSS, X/Twitter (Playwright browser scraping + API v2), arXiv API, custom web scrapers; implemented AI relevance filtering, source diversity penalties, and paper quota enforcement
> - Resolved complex LLM integration issues: model selection for structured JSON (chat vs reasoning), output token truncation, JSON parse fallback extraction, timestamp accuracy
> - Developed FastAPI web dashboard with archive, statistics, health monitoring, and cost tracking; REST API for reports and subscription management
> - Achieved 0% hallucination rate via deterministic URL whitelist validation + multi-layer quality gating; CI/CD automation with GitHub Actions
>
> **Tech Stack:** Python, FastAPI, DeepSeek, Playwright, Pydantic v2, Docker, GitHub Actions
