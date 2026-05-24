# Changelog

## v2.8 (2026-05-24) — 历史去重链路加固

### 问题
v2.7 修复后继续审查发现，历史去重仍有几个真实运行风险：
- 仓库里跟踪了旧的 `artifacts/drafts` 样例文件。CI clean checkout 时本地 artifacts 不为空，`load_recent_titles` 会提前返回，导致不再读取最近 GitHub issue 正文。
- LLM reranker 会在规则评分后重算 `rule_score`，可能把已被历史命中的旧事件重新抬高到候选前排。
- reranker prompt 要求判断发布时间和 URL，但候选输入没有传 `published_at/latest_seen_at`、URL、证据类型和 `already_reported` 标记。
- ResearchEditor 看到的历史列表混有 issue 标题、URL 和组合项，噪声偏大。
- 历史加载失败时 pipeline 静默 `pass`，线上很难发现去重退化。

### 修复

| # | 改动 | 文件 | 效果 |
|---|------|------|------|
| 35 | 新增 `load_recent_history`，本地 recent artifacts 与 GitHub issue 正文合并；本地 draft 按运行日期和 `window_days` 过滤，不再被旧样例截胡 | `history_checker.py` | CI 能稳定读取最近已发布日报正文 |
| 36 | `daily_report` 记录 `history_source/history_entry_count/history_error`，并写入 trace | `daily_report.py` | 历史加载失效可观测 |
| 37 | reranker 候选输入补充 `published_at`、`source_urls`、`evidence_type`、`confidence`、`already_reported` | `llm_reranker.py` | LLM 新鲜度和可靠性判断有依据 |
| 38 | reranker 对 `already_reported=true` 且无实质后续的事件强制封顶到 0.24 | `llm_reranker.py` | 旧闻不会在二次评分后重新浮上来 |
| 39 | ResearchEditor 历史上下文过滤 issue 标题和裸 URL，只保留去重后的事件标题 | `research_editor.py` | 编辑 agent 看到的历史更像“已报道事件列表” |

### 验证
- 本地编译：`python -m compileall agent -q`
- 定向测试：`python -m pytest tests/test_research_editor.py tests/test_llm_reranker.py tests/test_writer_output_schema.py -q`
- 全量测试：`python -m pytest -q`

## v2.7 (2026-05-24) — 编辑新鲜度与阅读体验优化

### 问题
对 2026-05-24 日报复盘后，发现自动编辑已经能生成完整日报，但离 Juya 式人工审核还有几个明显差距：
- 历史去重只读本地 draft 或 GitHub issue 标题；GitHub Actions 环境没有昨天 artifacts 时，只能看到“AI 日报 2026-05-24”这类 issue 标题，无法识别正文里已经发布过的事件。
- 已报道事件只做轻微扣分，DeepSeek 定价页、Google I/O 这类同 URL/同事件容易隔天再次进入主稿。
- 同一来源/同一厂商的多个更新会挤占版面，读起来像厂商更新列表，而不是编辑精选。
- Google AI Edge / LiteRT-LM / AI Edge Gallery 等同主题事件没有合并，容易拆成多条。
- 官方 X 发布源（如 StepFun）会因为社媒形态被误归到“前瞻与传闻”。
- 概览段落仍偏“条目堆叠”，且弱信号限定词容易污染整段；正文第二段有时只写泛泛影响，没有回答“谁受影响、下一步看什么”。

### 修复

| # | 改动 | 文件 | 效果 |
|---|------|------|------|
| 27 | `load_recent_titles` 增加 GitHub issue 正文解析，抽取概览/正文 Markdown 链接标题和 URL，并支持排除当天 issue | `history_checker.py` `daily_report.py` | CI 环境也能识别最近已发布条目，避免把今天自己的 issue 纳入历史 |
| 28 | 历史命中从“轻扣分”改为强降权：同标题或同 canonical URL 已报道且无实质后续时，分数封顶到 0.24 | `event_scorer.py` | 重复旧事件不再凭高基础分进入要闻 |
| 29 | Final selector 对 `already_reported` 且无实质后续的事件直接视为 stale background | `final_selector.py` | LLM 误选旧事件时，最终选择阶段仍能兜底过滤 |
| 30 | 同一来源默认最多 2 条，`must_include` 最多 3 条，并修复第一轮选择未累计 source count 的问题 | `final_selector.py` | 减少 Google/单一厂商 bundle 刷屏 |
| 31 | 新增 `google_ai_edge` story key，合并 LiteRT-LM、AI Edge Gallery、Gemma 端侧更新 | `final_selector.py` | 同主题多篇官方博客只保留最有代表性的一条 |
| 32 | 官方 X 发布源不再被默认归为前瞻；新增 StepFun/StepAudio 模型识别 | `section_classifier.py` `final_selector.py` `writer.py` | StepAudio 2.5 Realtime 等官方发布进入“模型发布” |
| 33 | Writer 自动重建 overview 为“今日主线 + 确认重点 + 前瞻信号”，并按整句取舍避免中文标题硬截断 | `writer.py` | 概览更像编辑判断，不再只是新闻清单 |
| 34 | Writer prompt 强制第二段回答“谁受影响”和“下一步看什么信号” | `prompts.yaml` | 正文分析更有读者价值，减少空泛句 |

### 验证
- 本地编译：`python -m compileall agent -q`
- 定向测试：`python -m pytest tests/test_research_editor.py tests/test_writer_output_schema.py -q`
- 全量测试：`python -m pytest -q`
- GitHub Actions：`Test` run `26355413960` 通过
- 提交：`4a1e3ab feat: improve daily editorial freshness`

## v2.3 (2026-05-21) — 覆盖大幅提升 & 架构优化

### 核心目标
解决日报漏抓热点新闻（如 Gemini 3.5 Flash 发布）和财经/财报数据（如 Nvidia 财报）的问题。

### P0 — 消除根因

| # | 改动 | 文件 | 效果 |
|---|------|------|------|
| 1 | 新增 `@GeminiApp` X 源 | `default.yaml` | Google 产品发布直连一手渠道 |
| 2 | 版本号保留（`3.5` ≠ `3 5`）+ 版本感知去重 | `event_clusterer.py` | "Gemini 3.5 Flash" 不再被合并到 "Gemini 2.5 Flash" |
| 3 | `_title_overlap` 字符集→SequenceMatcher + `_is_meaningful_update` 减罚（-0.12→-0.04） | `event_scorer.py` | 新版本发布不再被历史去重误伤 |
| 4 | 全链路 20+ 财报关键词（`earnings` `revenue` `财报` `营收` 等） | `curator.py` `event_scorer.py` `final_selector.py` `research_editor.py` `prompts.yaml` | 财经新闻不再被 relevance filter 过滤 |

### P1 — 拓展覆盖

| # | 改动 | 文件 | 效果 |
|---|------|------|------|
| 5 | 新增 CNBC + Reuters RSS 源（各 5→8 条/天） | `default.yaml` | 财经新闻首次入采集池 |
| 6 | "业界风向" cap 4→5 | `final_selector.py` `prompts.yaml` `default.yaml` | 多 1 个位置承载商业/政策新闻 |
| 7 | `_reader_utility` + `_impact_scope`：产品发布 (+0.15)、财经 (+0.10) boost；Bloomberg/CNBC/Reuters→trusted_media | `event_scorer.py` | 财务/产品新闻评分不再偏低 |

### P2 — 结构优化

| # | 改动 | 文件 | 效果 |
|---|------|------|------|
| 8 | breaking news boost：3 源以上 ×1.12，2 源+官方 ×1.06 | `event_scorer.py` | 多源确认的热点自动浮到前排 |
| 9 | `candidate_top_k` 40→60，`final_max_items` 22→25 | `default.yaml` | 高峰日不再因截断丢失重要事件 |
| 10 | 7 板块拆分：`资本动向`（融资/财报/IPO）+ `产业风向`（政策/并购/人事） | `final_selector.py` `research_editor.py` `writer.py` `prompts.yaml` `critic.py` `mock_provider.py` | 财报不再跟政策挤同一个板块 |

### Bug 修复：X Cookie Context 泄漏

**根因**：`XCookieAdapter._get_page()` 每次创建新 BrowserContext，但 `fetch()` 只关 Page 不关 Context。24 个 X 源累积 24 个 Context → 浏览器资源耗尽 → Windows IOCP `GetQueuedCompletionStatus` 卡死。

**修复**：
- `_get_page()` 返回 `(context, page)` 元组
- `fetch()` 的 `finally` 同时关闭两者
- 移除"代理失败后 fallback 直连"逻辑（GFW 环境直连 x.com 永远超时）

**效果**：X 源抓取从 35s/源 → 12-15s/源

### 覆盖效果

| 维度 | v2.2 | v2.3 |
|------|:--:|:--:|
| 重大头条覆盖 | ~30% | ~45% |
| 财经/财报条目 | 0 | 2-3 |
| 资本动向板块 | 不存在 | 独立 cap=3 |
| X 源稳定性 | 偶发卡死 | Context 泄漏已修复 |

### 后续优化（v2.4）

| # | 改动 | 文件 |
|---|------|------|
| 11 | CNBC/Reuters 权重 0.95/1.0→1.1，max_items 5→8 | `default.yaml` |
| 12 | ResearchEditor prompt 将"重要财报/融资/IPO"提升为第 2 优先级 | `research_editor.py` |
| 13 | 资本动向保底 min 2 条 | `final_selector.py` |
| 14 | `insider_media` content_type 权重 1.05→1.15 | `default.yaml` |

## v2.4 (2026-05-21) — LLM 双重评分把关

### 问题
规则评分对财经/财报/产品发布类新闻评分偏低（因其偏技术维度），即使 CNBC/Reuters 源已采集到数据，LLM ResearchEditor 仍可能不选。

### 方案
在规则评分和 ResearchEditor 之间插入 LLM Re-rank 步骤：
1. 规则评分 → top 60 事件
2. **LLM 热度评估** → 评估每个事件真实世界热度（1-10 分）
3. 组合分数（rule_score ×0.35 + llm_score ×0.65）→ 重新排名
4. ResearchEditor → 编辑选择

### 改动

| # | 改动 | 文件 |
|---|------|------|
| 15 | 新增 `llm_reranker.py` — LLM 热度双重评分模块 | `agent/agents/llm_reranker.py` |
| 16 | 管线插入 rerank 步骤（规则评分后、Editor 前） | `agent/pipelines/daily_report.py` |

### 效果预期
- 财报/财经新闻（Nvidia 季度营收、Anthropic 盈利）被 LLM 识别为高热度，组合分数提升 → ResearchEditor 更容易选中
- 增量成本：1 次额外 LLM 调用（~12K input + ~2K output tokens）

## v2.5 (2026-05-22) — 修复论文精选准入漏洞 (Issue #40)

### 问题
non-academic Anthropic 产品更新（Claude Code changelog、合作伙伴案例研究）被 Writer/ResearchEditor 放进"论文精选"板块。

### 根因
论文精选的准入/准出逻辑不对称：
- 有"非论文赶出去"（`论文精选` → `产业风向`）
- 没有"论文拉回来"（`模型前沿` → `论文精选`）
- `_guess_section` 先用关键词再用 arXiv 源类型检查，ArXiv 论文可能被"模型""benchmark"等关键词拐进模型前沿

### 修复

| # | 改动 | 文件 |
|---|------|------|
| 17 | `_parse_and_validate` 增加反向检查：arXiv/HF URL 条目在非头条非论文板块 → 移到论文精选 | `research_editor.py` |
| 18 | `_guess_section` 将 arXiv 源类型检查提到所有关键词之前 | `final_selector.py` |
| 19 | Writer 硬性要求第 7 条：论文精选只能放 arxiv_top_venue/hf_daily_papers 条目 | `prompts.yaml` |
| 20 | ResearchEditor 论文精选描述明确排除"软件 changelog、技术博客、案例研究" | `research_editor.py` `prompts.yaml` |

## v2.6 (2026-05-23) — Juya-style 自动编辑审核

### 问题
日报已经能自动发布，但选题仍像“规则聚合”：论文和大厂官方博客偏重，API 价格、国产模型、开发工具等中文读者高价值信息容易排不到前面；未确认融资和 A/B 测试也会混入主板块。

### 改动

| # | 改动 | 文件 | 效果 |
|---|------|------|------|
| 21 | 7 板块切换为 `要闻/模型发布/开发生态/技术与洞察/产品应用/行业动态/前瞻与传闻` | `section_classifier.py` `final_selector.py` `writer.py` `research_editor.py` `prompts.yaml` | 接近 Juya 的人工编辑分组 |
| 22 | 移除论文 min 5 和资本 min 2 的机械保底，改为核心板块软覆盖 | `final_selector.py` | 不再为了填栏目牺牲选题质量 |
| 23 | LLM reranker 加入中文读者和可行动信息偏好 | `llm_reranker.py` `event_scorer.py` | DeepSeek/Qwen/智谱、价格、API、工具更新更容易进入前排 |
| 24 | 新增传闻/前瞻隔离规则，弱信号最多 2 条 | `section_classifier.py` `final_selector.py` `research_editor.py` | 未确认消息不再污染主新闻判断 |
| 25 | 收紧相关链接聚合，移除过宽 `codex` story key | `final_selector.py` | 减少无关参考链接混入正文 |
| 26 | Writer fallback 输出人类可读证据说明，避免裸露 tier/evidence 元数据 | `writer.py` | 内容更像编辑稿，少像调试信息 |
