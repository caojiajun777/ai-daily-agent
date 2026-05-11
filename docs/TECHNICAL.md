# AI 日报 — 技术文档

## 项目概述

全自动 AI 科技日报生成系统。每日从 30+ 全球化信息源采集 AI 领域最新资讯，经 7 个角色化 Agent 协作流水线处理后，生成结构化的中文深度日报，发布至 GitHub Issues 并推送 PDF 至订阅者邮箱。

**GitHub:** [github.com/caojiajun777/ai-daily-agent](https://github.com/caojiajun777/ai-daily-agent)

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    AI 日报自动化流水线                        │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Collector│→│ Curator  │→│  Writer  │→│  Critic  │   │
│  │ 30+ RSS  │  │ 去重/排序 │  │ DeepSeek │  │ 幻觉检测 │   │
│  │ X/API/Web│  │ 相关性   │  │ V4 Pro   │  │ 语义去重 │   │
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
| `rss.py` | RSS/Atom | 支持标准 RSS 2.0 和 Atom 1.0 格式，自动解析标题/摘要/发布时间，HTML 标签清洗 |
| `x_adapter.py` | X/Twitter API v2 | Bearer Token 认证，支持 official/kol/media 账号分类，按时间窗口过滤推文，排除转推和回复 |
| `aihot_adapter.py` | Web Scraper | 解析 AI HOT Daily 页面，提取结构化新闻条目及原始来源标注，自动映射到本地信源 ID |
| `base.py` | 工厂模式 | 统一 SourceAdapter 协议，通过 YAML 配置驱动适配器实例化 |

**关键指标：**
- 30+ 信源接入（14 RSS + 16 X + 1 Web Scraper）
- 信源覆盖：OpenAI、Anthropic、Google DeepMind、Meta AI、百度、阿里云、阶跃星辰等
- 单次采集 80-110 条原始条目
- 语言：中文 + 英文双语覆盖

### 2. 内容策展层 (`agent/agents/curator.py`)

**处理流程：**
1. 标题规范化（Unicode 归一化，标点清洗）
2. 精确去重（标题哈希 + URL 集合）
3. AI 相关性过滤（中英文双关键词库，60+ 关键词匹配）
4. 多维打分：`score = source_weight × recency × relevance_boost × diversity_penalty`

**关键算法：**
- **时效性衰减**：指数衰减模型，72 小时半衰期
- **来源多样性惩罚**：同一源第 4 条 ×0.85，第 7 条 ×0.60，防止单源霸榜
- **通用源隔离**：IT之家等泛科技源需 ≥2 个英文关键词 或 ≥1 个中文关键词才通过

### 3. LLM 写稿层 (`agent/agents/writer.py`)

- **模型**：DeepSeek V4 Pro (via OpenAI-compatible SDK)
- **输出格式**：6 个固定板块（要闻/模型发布/开发生态/产品应用/技术与洞察/行业动态）
- **每条条目**：150-350 字深度分析 + 2-4 条要点提炼
- **容错机制**：
  - Think block 自动剥离（DeepSeek 推理模型特性）
  - JSON 修复引擎（尾逗号删除、缺失逗号补全、字符串拼接修复）
  - 日期覆盖保护（不信任 LLM 返回的日期字段）

### 4. 质量审核层 (`agent/agents/critic.py`)

**两层审核架构：**

| 层级 | 类型 | 检查项 |
|------|------|--------|
| 第一层 | 确定性审核 | URL 白名单校验、板块/条目数量检查、禁止短语检测、重复标题检测 |
| 第二层 | 语义去重 | 单次 LLM 调用检测跨板块同一事件重复（high/medium/low 三级严重度） |

**质量门控（Publish Gate）：** 必须通过所有检查才允许发布，不通过标记 `needs_human_review`

### 5. 自动修复层 (`agent/agents/repairer.py`)

- 触发条件：语义去重检测到 high/medium 级别重复
- 修复操作：移除/替换重复条目
- 防注入：替换 URL 必须在策展白名单内，否则拒绝替换

### 6. 配图提取层 (`agent/tools/image_extractor.py`)

**双层提取架构：**

| 层级 | 技术 | 适用场景 |
|------|------|---------|
| 第一层 | HTTP og:image | 传统服务端渲染网站（WIRED、MIT Tech Review） |
| 第二层 | Playwright headless | JS SPA 网站（qbitai、OpenAI） |

**智能评分：** 图片文件名质量 + UUID 检测 + CDN 路径识别 + 来源优先级 + 噪声过滤（Logo/QR 码/头像/占位图）

**X/Twitter 支持：** 注入完整 session cookies 绕过登录墙，提取推文媒体图片（`pbs.twimg.com/media`）

### 7. 发布层 (`agent/tools/issue_publisher.py` & `agent/agents/issue_publisher.py`)

- GitHub Issues API 发布
- 重复检测（30 天内标题匹配）
- 强制模式（`--force`）绕过重复检测但不绕过质量门控
- 发布预览（`--dry-run`）本地验证

### 8. PDF + 邮件层 (`agent/tools/pdf_emailer.py`)

- MD → HTML → PDF：Playwright Chromium 渲染，A4 格式，中文字体支持
- SMTP 邮件：QQ 邮箱 SMTP SSL（465 端口），HTML 正文 + PDF 附件
- 订阅管理：Issue 评论自动订阅/退订 + 一次性邮箱黑名单 + 注入检测
- 多信源加载：GitHub Issue 评论 → `subscribers.txt` → `default.yaml`

### 9. Web 看板 (`agent/web/`)

| 路由 | 功能 |
|------|------|
| `/` | 首页 — 最新日报全文 + 系统概览统计 |
| `/archive` | 归档 — 所有历史报告列表（日期/条目数/状态） |
| `/stats` | 统计 — 信源分布柱状图 + Token 用量 + 14 天流水线历史 |
| `/monitor` | 监控 — 信源健康表 + 流水线成功率 + 成本追踪 + 告警 |
| `/report/{date}` | 单篇报告查看 |

**REST API：**
- `GET /api/reports` — 结构化日报数据
- `GET /api/reports/{date}` — 单日报 JSON
- `GET /api/stats` — 聚合统计
- `GET /api/monitor` — 健康检查 JSON
- `POST /api/subscribe?email=xxx` — 订阅
- `POST /api/unsubscribe?email=xxx` — 退订

### 10. 监控层 (`agent/tools/monitor.py`)

- **信源健康**：定期检查所有 RSS feed 新鲜度，按 ok/stale/dead 分级
- **流水线健康**：14 天历史成功率、Token 用量、成本统计
- **告警**：连续失败/信源失效自动邮件通知

### 11. 基础设施层 (`agent/harness/`)

| 组件 | 功能 |
|------|------|
| `state.py` | 流水线状态机（pending/running/ok/failed/needs_human_review/skipped），每阶段独立状态 |
| `trace.py` | Append-only JSONL 日志，含 prompt hash、token 估算（CJK 感知）、LLM 调用追踪 |
| `budget.py` | Token 预算管理，分阶段归属，硬上限保护 |
| `tools.py` | ToolRegistry + JSON-Schema 参数验证（预留工具调用能力） |
| `context.py` | 滚动窗口消息裁剪，单条消息字符上限 |

### 12. LLM 抽象层 (`agent/llm/`)

- `LLMProvider` Protocol：统一接口，支持 DeepSeek / Anthropic / OpenAI-compatible 切换
- `DeepSeekProvider`：OpenAI-compatible SDK 封装，Fail-fast 模型验证，自动重试（max_retries=2）
- `MockLLMProvider`：确定性 mock，离线测试用，支持自定义 responder 函数
- 深度集成 tracer、budget tracker

---

## 技术栈

| 类别 | 技术 | 用途 |
|------|------|------|
| 语言 | Python 3.12+ | 全部后端逻辑 |
| LLM | DeepSeek V4 Pro | 日报写作 + 语义去重 + 信源发现 |
| Web 框架 | FastAPI + Jinja2 | Web 看板 + REST API |
| 数据采集 | feedparser, httpx, X API v2 | RSS/API/Web 多协议采集 |
| 浏览器自动化 | Playwright (Chromium/Edge) | JS 渲染页面配图 + PDF 生成 |
| 数据建模 | Pydantic v2 | 全流水线数据验证 |
| 测试 | pytest (138 tests) | 离线 mock 全覆盖 |
| CI/CD | GitHub Actions | 定时 + 订阅 + 健康检查 + 测试 |
| 容器化 | Docker Compose | 一键部署 |
| 监控 | 自研 monitor.py | 信源/流水线/成本追踪 |
| 配置 | YAML + 环境变量 | 分层配置管理 |

---

## 关键指标

### 系统规模

| 指标 | 数值 |
|------|------|
| 信源数量 | 30+ (14 RSS + 16 X + 1 Scraper) |
| Agent 数量 | 7 (Collector/Curator/Writer/Critic/SemanticDup/Repairer/Publisher) |
| 代码文件 | 128 个 |
| 测试用例 | 138 个 |
| CI Workflows | 4 个 (日报/订阅/健康检查/测试) |

### 运行指标

| 指标 | 典型值 |
|------|--------|
| 单次采集量 | 80-110 条 |
| 策展后数量 | 20 条 |
| 最终条目 | 16-22 条 |
| LLM 调用次数 | 2 次/天 (write + semantic dup) |
| Token 消耗 | ~7,000 input + ~1,500 output |
| 单次成本 | ~$0.004 |
| 日报板块 | 6 个 |
| 信源语言 | 中文 + 英文 |

### 质量指标

| 指标 | 数值 |
|------|------|
| 标题唯一率 | 100% |
| 幻觉链接率 | 0% (URL 白名单强制校验) |
| 禁止短语命中率 | 0% |
| 语义重复率 | 0% (自动修复) |
| 测试覆盖率 | 核心路径全覆盖 |

---

## 简历可用表述

### 中文版

> **AI 日报全自动生成系统** | Python, FastAPI, DeepSeek LLM, GitHub Actions
>
> - 独立设计并实现了 7-Agent 协作流水线架构，覆盖数据采集、策展打分、LLM 深度写稿、多层质量审核、语义去重与自动修复、发布推送全链路
> - 构建了 30+ 全球化信息源聚合系统，支持 RSS、X/Twitter API v2、Web Scraper 多协议接入，含 AI 相关性过滤、来源多样性惩罚等策展算法
> - 基于 FastAPI 构建 Web 看板及 REST API，含日报归档、信源分布统计、系统健康监控；Docker Compose 一键部署
> - 实现 Playwright headless 配图自动提取（双层架构：HTTP og:image + JS 渲染回退）及 PDF 邮件推送
> - 编写 138 个 pytest 离线单测；设计多层安全验证（URL 白名单校验、XSS/SQL 注入检测、一次性邮箱黑名单）；全 GitHub Actions 自动化编排

### 英文版

> **AI Daily Report — End-to-End Automated News Generation System**
>
> - Architected a 7-Agent collaborative pipeline (Collect → Curate → Write → Critique → Semantic Dedup → Repair → Publish) processing 30+ global sources into structured Chinese daily reports
> - Built multi-source aggregation system supporting RSS, X/Twitter API v2, and custom web scrapers; implemented AI relevance filtering, source diversity penalties, and exponential recency decay scoring
> - Developed FastAPI web dashboard with archive, statistics, and health monitoring views; exposed REST API for reports, stats, and subscription management
> - Implemented two-tier image extraction (HTTP og:image + Playwright headless fallback for JS SPAs) with intelligent scoring and noise filtering; integrated SMTP email delivery with PDF generation
> - Achieved 100% hallucination-free output via deterministic URL whitelist validation; 138 offline pytest cases; full CI/CD automation with GitHub Actions
>
> **Tech Stack:** Python, FastAPI, DeepSeek LLM, Playwright, Pydantic, Docker, GitHub Actions
