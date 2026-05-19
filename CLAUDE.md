# CLAUDE.md

## 项目概述

全自动多 Agent AI 日报系统。每日定时（UTC 04:00 = 北京时间 12:00）从多个信息源采集 AI 行业动态，经多 Agent 协作去重、评分、筛选、撰写、审核后，自动发布为 GitHub Issue 并邮件推送 PDF。

当前版本：**v2.2** — LLM 筛选模型切换到 deepseek-chat，X/Twitter 源在 CI 直接可访问，6 板块新分类，论文独立板块。

## 核心架构

### 入口

- CLI: `agent/cli.py`（10 个子命令，含 `admit-sources`）
- 主管线: `agent/pipelines/daily_report.py`（编排所有 Stage）
- Web 面板: `agent/web/app.py`（FastAPI + Jinja2）
- CI/CD: `.github/workflows/daily_run.yml`（定时触发）
- 测试 CI: `.github/workflows/test.yml`（push/PR 触发）
- 自动纳新: `agent/tools/auto_admit.py`（scout 发现 → 低权重写入配置）

### Agent 角色及调用链

```text
SourceDiscoverer → Collector → EventClusterer → EventScorer
  → HistoryChecker → ResearchEditor → FinalSelector
  → Writer → SemanticDuplicateCritic → Repairer
  → Critic → IssuePublisher
```

### LLM 模型策略（关键）

| Stage | 模型 | 原因 |
|-------|------|------|
| **ResearchEditor** | **deepseek-chat** | v4-pro 推理模型输出 0 条；chat 生成结构化 JSON 稳定 |
| **Writer** | **deepseek-chat** | v4-pro 生成格式损坏的 JSON（`Expecting ',' delimiter`） |
| SemanticDuplicate | deepseek-v4-pro | 简单输出，v4-pro 可用 |
| Scout LLM 发现 | **deepseek-chat** | 同 ResearchEditor，需要结构化 JSON |

**禁止在需要结构化 JSON 输出的 Stage 使用 v4-pro 推理模型。** v4-pro 花几百秒"思考"但输出极短或格式损坏。

### ResearchEditor 调试备忘

- `max_output_tokens=6144`（原来 3072）——16-24 条完整编辑决策需要 ~15000 字符，3072 tokens 在 ~9100 字符处截断 JSON 导致解析失败
- `_parse_and_validate` 新增 `{...}` fallback：deepseek-chat 可能在 JSON 外加对话文本，从最外层 `{}` 提取
- 可通过 trace 中的 `editor_parse_result` 事件查看 `raw_len`, `raw_suffix`, `notes` 诊断解析问题
- 本地测试方法：从 artifact 提取 curated items → 构造 EventCluster → 调用真实 API → 检查 `_parse_and_validate` 输出

### 日报 6 板块（v2.2）

```
1. 今日头条  cap=3  Headlines
2. 模型前沿  cap=4  Model Frontier       ← 模型/架构/Benchmark
3. 工具与开源 cap=3  Tools & Open Source   ← SDK/API/框架/定价
4. 论文精选  cap=5  Paper Picks           ← arXiv/HF 论文专属
5. 产品落地  cap=3  Launchpad             ← 产品/功能/应用
6. 业界风向  cap=4  Industry Watch        ← 融资/政策/人事
```

分类视角从"这是什么类型的内容"转变为"这对读者意味着什么"。论文有独立板块，不再是"技术与洞察"大杂烩。

相关配置位置：
- `final_selector.py`: `section_caps`, `section_order`, `_guess_section`
- `research_editor.py`: `_RESEARCH_EDITOR_PROMPT` 中的 section 选项
- `writer.py`: `_SECTION_SUBTITLES`
- `prompts.yaml`: Writer 系统提示中的 section 定义
- `repairer.py`: 优先保留板块引用

### 目录结构

```text
agent/
├── agents/        # 各 Agent 角色实现
│   ├── collector.py            # 信息采集
│   ├── curator.py              # 筛选编排（3 种模式）+ _select_with_paper_quota
│   ├── event_clusterer.py      # 事件聚合（URL/标题/相似度聚类）
│   ├── event_scorer.py         # 规则评分
│   ├── history_checker.py      # 历史去重
│   ├── research_editor.py      # LLM 研究编辑（默认筛选模式）
│   ├── final_selector.py       # 最终筛选 + 论文配额 + section 分配
│   ├── writer.py               # Markdown 日报撰写
│   ├── critic.py               # 确定性评审
│   ├── semantic_duplicate_critic.py  # LLM 语义去重
│   ├── repairer.py             # 修复重复项
│   ├── issue_publisher.py      # GitHub Issue 发布 + Publish Gate
│   ├── trend_analyzer.py       # 趋势分析
│   ├── trend_validator.py      # 趋势验证
│   ├── source_discoverer.py    # LLM 来源发现
│   ├── source_scout.py         # 三通道统一来源搜索
│   └── source_diffuser.py      # 社交图/内容链扩散
├── llm/           # LLM Provider
│   ├── factory.py              # 统一构建入口 build_provider()
│   ├── anthropic_provider.py
│   ├── deepseek_provider.py    # deepseek-chat + deepseek-v4-pro
│   ├── openai_compatible_provider.py
│   ├── qwen_provider.py
│   └── mock_provider.py
├── sources/       # 信息源适配器
│   ├── base.py                 # RawItem, SourceSpec 基类
│   ├── rss.py                  # RSS 源
│   ├── arxiv_adapter.py        # arXiv API（免 key）
│   ├── x_adapter.py            # X/Twitter API（需 X_BEARER_TOKEN）
│   ├── x_cookie_adapter.py     # X Playwright 抓取（CI 免代理直连）
│   ├── aihot_adapter.py        # AI Hot 中文源
│   ├── arxiv_stub.py           # arXiv 轻量桩
│   ├── hn_stub.py              # HackerNews 桩
│   └── github_stub.py          # GitHub Trending 桩
├── tools/         # 工具函数
│   ├── auto_admit.py           # Scout 发现自动低权重纳新
│   ├── evidence_fetcher.py     # 证据获取
│   ├── image_extractor.py      # og:image 提取
│   ├── pdf_emailer.py          # PDF 生成 + 邮件推送
│   ├── trend_metrics.py        # 趋势指标
│   └── vision_enricher.py      # 视觉增强
├── pipelines/     # 管线编排
├── harness/       # 上下文、预算、追踪、回放
├── schemas.py     # Pydantic 数据模型
├── configs/       # YAML 配置文件
└── web/           # FastAPI Web 面板 + Jinja2 模板
tests/             # 测试文件（含 test_research_editor, test_trends 等）
```

### 配置

- 主配置: [agent/configs/default.yaml](agent/configs/default.yaml) — 数据源、LLM、筛选、发布参数（56 个源）
- 提示词: `agent/configs/prompts.yaml` — writer/critic system/user 模板
- 环境变量: `.env`（从 `.env.example` 复制）— API key、GitHub Token、邮箱等
- Provider 工厂: [agent/llm/factory.py](agent/llm/factory.py) — `build_provider(name, model, **kwargs)`

### 筛选模式

`curation.mode` 控制筛选策略：

| mode | 说明 |
|------|------|
| `research_editor` | **默认**：Cluster → Score → HistoryCheck → ResearchEditor(LLM) → FinalSelect |
| `rules_only` | 纯规则筛选，不调用 LLM |
| `legacy_llm_scoring` | 旧版 LLM 评分（需显式启用 `legacy_llm_scoring_enabled`） |

ResearchEditor 回退机制：当 LLM 选择的条目 < `final_min_items`(16) 时，`final_selector` 按 rule_score 填充差额，并强制执行论文配额（min 5 篇 arXiv/HF papers）。

### 论文配额机制

两个文件都有保底逻辑：
- `curator.py`: `_select_with_paper_quota(scored, max_items, min_papers=5)` — 旧版筛选
- `final_selector.py`: `_MIN_PAPERS = 5` — ResearchEditor 回退 + section 分配

论文判定依据 `source_types` 包含 `"arxiv"` 或 `source_names` 包含 `"hf_daily_papers"`。

### Publish Gate 规则

发布门禁在 `issue_publisher.py` 的 `evaluate_publish_gate()`：

| 条件 | 阻塞？ | 说明 |
|------|--------|------|
| draft 为空 | 阻塞 | |
| 总条目 < minimum_items(3) | 阻塞 | |
| critic verdict != "pass" | 阻塞 | |
| eval issues > 0 | 阻塞 | 含幻觉 URL |
| 语义重复 high/medium | 条件阻塞 | repair 成功或相同URL 则降级为 warning |
| repair 失败 | 条件阻塞 | 所有重复为相同URL 则不阻塞 |

`--force` 跳过 gate + 重复检查；`--force-dup` 只跳过重复检查。

## CLI 命令

### 运行日报

```bash
python -m agent.cli run                          # 默认 provider（deepseek）
python -m agent.cli run --provider mock          # Mock 模式（无 API 调用）
python -m agent.cli run --provider deepseek --model deepseek-chat
python -m agent.cli run --date 2026-05-18        # 指定日期
python -m agent.cli run --skip-model-check       # DeepSeek API 网关屏蔽 list 时
```

### 评估

```bash
python -m agent.cli eval --run-id 2026-05-18
```

### 发布 Issue

```bash
python -m agent.cli publish-issue --run-id 2026-05-18 --dry-run     # 预览
python -m agent.cli publish-issue --run-id 2026-05-18 --confirm     # 真发
python -m agent.cli publish-issue --run-id 2026-05-18 --confirm --force      # 跳过 gate + 重复
python -m agent.cli publish-issue --run-id 2026-05-18 --confirm --force-dup  # 只跳过重复
```

### 趋势分析

```bash
python -m agent.cli trends --days 7
python -m agent.cli trends --multi-window 4,7,14,30
```

### 来源发现

```bash
python -m agent.cli discover-sources --topic chinese-ai-models
python -m agent.cli scout --topic broad --run-id 2026-05-18
python -m agent.cli diffuse --run-id 2026-05-18
python -m agent.cli admit-sources --scout-report <path> --dry-run   # 预览纳新
```

### 其他

```bash
python -m agent.cli replay --run-id 2026-05-18          # 回放 trace
python -m agent.cli send --run-id 2026-05-18            # 生成 PDF 并邮件推送
python -m agent.cli verify-gitblog --issue-number 42    # 验证已发布 Issue
```

## 常用命令

### 运行测试

```bash
pytest tests/ -v
pytest tests/test_issue_publisher.py -v
pytest tests/test_research_editor.py -v
pytest tests/test_trends.py -v
pytest tests/test_source_discoverer.py -v
pytest tests/test_source_scout.py -v
```

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动 Web 面板

```bash
uvicorn agent.web.app:app --reload --port 8000
```

## 技术约定

### Python & 架构
- Python 3.10+
- Agent 接口: `process(state)` 接收/返回 State 对象；新 Agent 在 `agent/agents/__init__.py` 注册
- LLM 调用统一通过 [agent/llm/factory.py](agent/llm/factory.py) 的 `build_provider()`，禁止直接实例化
- 数据模型定义在 [agent/schemas.py](agent/schemas.py)，修改后检查管线兼容性
- 管线 Stage 状态: pending → running → ok / failed / needs_human_review / skipped

### LLM 模型选择（重要）
- **需要结构化 JSON 输出的 Stage 必须用 deepseek-chat，禁止用 v4-pro 推理模型**
- ResearchEditor `max_output_tokens=6144`（3000 不够装 16-24 条完整 JSON）
- Writer 也需要 deepseek-chat（v4-pro 输出 `Expecting ',' delimiter`）
- Budget 硬限制: 默认 200K input tokens, 30K output tokens, 40 次 LLM 调用，超限抛异常
- 通过 `build_provider("deepseek", model="deepseek-chat", skip_model_check=True, request_timeout_s=N)` 创建独立 provider

### X/Twitter 源（XCookieAdapter）
- 使用 Playwright 抓取 `x.com/{username}` 页面
- CI 环境（美国 runner）**不需要代理**：无 `X_PROXY` 环境变量时自动直连
- 本地中国环境需要 `X_PROXY` 或 `HTTPS_PROXY` 环境变量
- 从 HTML 提取真实推文链接 `/{username}/status/{tweet_id}` 和 `<time datetime="...">` 时间戳
- 旧推文通过 `max_age_hours` 正确过滤（之前 `published_at` 写死 `datetime.now()` 导致置顶旧闻通过）
- `published_at` 提取失败时 fallback 为 `datetime.now()`

### 信息源注意事项
- **Anthropic 没有公开 RSS feed**（`/news/rss.xml` 返回 404）。覆盖来自 X @AnthropicAI、TechCrunch AI RSS、The Verge AI RSS
- **36kr RSS** (`36kr.com/feed?cid=330`) URL 格式特殊，Writer 容易幻觉链接。当前权重 0.4（观察中）
- **arXiv** 每天抓 100 条（`max_items: 100`），`top_venue_only: false` 不过在客户端用 `_mentions_top_venue` 过滤
- HuggingFace Daily Papers 是 RSS（`huggingface.co/papers/feed.xml`），source_type 为 "rss" 不是 "arxiv"

### Scout 源发现系统
- 三通道：LLM 语义发现 + 内容链扩散 + 社交图扩散
- LLM 通道需 deepseek-chat（非推理模型）
- 内容链扩散需要 `collected_items` 作为种子——pipeline 在 collect 阶段保存 `artifacts/collected/{date}.json`
- Scout CLI `--run-id` 优先读取 `collected/{date}.json`（全量），fallback 到 `curated/{date}.json`（精选 16 条）
- `admit-sources` 命令读 scout 报告 → 去重 → 追加到 `default.yaml`（低权重 0.55）
- 当前 CI 只做 dry-run（`--dry-run`），不自动纳新

### Publish Gate & 重复处理
- 语义重复检测后执行修复（repairer），修复成功则门禁不再阻塞相同重复
- 相同 URL 的语义重复（"相同URL""同一推文"）降级为 warning，不阻塞发布
- `--force-dup` 只跳过重复检查，不跳过质量门禁；`--force` 跳过一切

### 其他
- `.env` 在 gitignore 中，CI 通过 GitHub Secrets 注入
- 禁止的语言模式: "作为AI"、"I cannot"、"Lorem ipsum"
- `artifacts/` 目录在 gitignore 中，不在 repo 中
- 测试中的 RSS 源用本地临时文件（`tmp_path / "test_feed.xml"`）代替网络请求

## 环境变量

| 变量 | 用途 | 必需 |
|------|------|------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | 是 |
| `GITHUB_PUBLISH_TOKEN` | 发布 Issue 的 PAT（映射自 `secrets.PUBLISH_TOKEN`） | 发布时 |
| `PUBLISH_REPO` | 目标仓库（owner/repo） | 发布时 |
| `PUBLISH_ISSUE_LABELS` | Issue 标签（逗号分隔） | 否 |
| `X_BEARER_TOKEN` | X/Twitter API v2 Token | 否（无则 X 源抓取降级） |
| `X_PROXY` | X Playwright 抓取代理（中国环境需要） | 否（CI 自动直连） |
| `SMTP_USER` / `SMTP_PASSWORD` | 邮件推送凭证 | 发送时 |
| `EMAIL_SUBSCRIBERS` | 邮件订阅列表（逗号分隔） | 发送时 |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | 备用 Provider | 否 |

**CI Secret 名注意**：workflow 文件中 GitHub Secret 叫 `PUBLISH_TOKEN`（非 `GITHUB_PUBLISH_TOKEN`），通过 `GITHUB_PUBLISH_TOKEN: ${{ secrets.PUBLISH_TOKEN }}` 映射。

## 当前状态

- v2.2 架构：ResearchEditor + Writer 均用 deepseek-chat，X 源 CI 直连
- 6 板块新分类（今日头条/模型前沿/工具与开源/论文精选/产品落地/业界风向）
- 论文配额（min 5 篇）在 curator 和 final_selector 双路径保障
- CI 每日 12:00 CST 自动运行（`daily_run.yml`）
- 订阅推送 CI 独立触发（`subscribe.yml`）
- 测试 CI 在 push/PR 时自动运行（`test.yml`）
- 46 个有效信息源：23 X/Twitter + 20 RSS + arXiv + AI Hot + 36kr
- Scout 源发现每日运行，dry-run 模式（结果在 artifact）
- 已知改进空间：ResearchEditor LLM 仍偶尔选 0 条（需监控 editor_parse_result trace），内容链扩散通道偶有网络波动
