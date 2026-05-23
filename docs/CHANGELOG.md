# Changelog

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
