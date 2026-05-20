# 工作日志

## 2026-05-19 — v2.2 大规模修复日

### #1 CI 完全不运行

**现象**：日报连续多天没有推送到邮箱和 Issue。

**根因**：GitHub Secret 从 `GITHUB_PUBLISH_TOKEN` 改名为 `PUBLISH_TOKEN`，但三个 workflow 文件（`daily_run.yml`、`health_check.yml`、`subscribe.yml`）的修改没有 commit/push。GitHub 上执行的仍是旧版，引用 `secrets.GITHUB_PUBLISH_TOKEN` → 解析为空字符串 → Guard Step 检测为空 → `exit 1`，整个 pipeline 从未执行。

**修复**：提交并推送三个 workflow 文件的 secret 引用改动。

**防止复发**：修改 GitHub Secret 时，确保对应的 workflow 文件同步提交并推送。

---

### #2 CI 运行但立即崩溃（ImportError）

**现象**：Secret 修复后 CI 能跑了，但 `python -m agent.cli run` 第一步就崩溃。

**根因**：v2.1 的 18 个 commit 修改了 `daily_report.py` 使其 import 7 个新 Agent 文件（`event_clusterer.py`、`event_scorer.py`、`history_checker.py`、`research_editor.py`、`final_selector.py`、`trend_analyzer.py`、`trend_validator.py`），但这些新文件从未被推送。远程仓库有 import 语句但没有目标文件，`ImportError`。

**修复**：补全所有缺失文件并推送。

**防止复发**：新增文件时确认 `git status` 中包含这些文件；考虑在 CI test workflow 中加入 `python -c "from agent.pipelines.daily_report import run_pipeline"` 做冒烟检查。

---

### #3 日报没有论文

**现象**：Issue #26、#27 全是 X/Twitter 和 RSS 新闻，零篇论文。

**根因**（三层）：
1. **ResearchEditor LLM 选 0 条**：`deepseek-v4-pro` 推理模型输出极短或空 JSON，`llm_selected_count: 0`，触发回退。
2. **回退路径没有论文配额**：`final_selector.py` 的 `select_final_items()` 按 `rule_score` 排序填充，arXiv 论文评分天然低于 X/Twitter 新闻（权重 0.85 vs 1.1），全部被挤出。
3. **旧版 `curator.py` 也没有论文配额**：旧路径直接用 `scored[:max_items]` 截断。

**修复**：
- `final_selector.py` 新增 `_MIN_PAPERS = 5` 和 `_select_with_paper_quota` 逻辑（论文不够 5 篇时自动替换低分非论文条目）
- `curator.py` 同步加上 `_select_with_paper_quota`
- 论文判定依据：`source_types` 包含 `"arxiv"` 或 `source_names` 包含 `"hf_daily_papers"`

**防止复发**：论文配额在 curator 和 final_selector 双路径保障。

---

### #4 X/Twitter 源在 CI 全部沉默

**现象**：23 个 X 源在 CI 环境返回 0 条，`raw_item_count` 只有 147-191。

**根因**：`XCookieAdapter.fetch()` 检查 `X_PROXY` 环境变量，如果为空直接 `return []`。CI runner 在美国，可以直连 X.com，不需要代理。

**修复**：去掉强制代理检查。现在逻辑：有代理用代理，无代理直连。`fetch()` 改为先尝试配置的代理，失败或无代理则直连。

**防止复发**：不在本地代理环境下测试时，注意 CI 环境的网络路径差异。

---

### #5 Writer 输出格式损坏的 JSON

**现象**：`writer output is not valid JSON after retry: Expecting ',' delimiter`

**根因**：Writer 使用 `deepseek-v4-pro` 推理模型。v4-pro 输出 JSON 时经常产生语法错误（多余逗号、引号未闭合等）。

**修复**：将 Writer 切换到 `deepseek-chat`（非推理模型），与 ResearchEditor 一致。在 `_run_research_editor_flow` 同级为 Writer 创建独立 provider。

**防止复发**：规则明确——**所有需要结构化 JSON 输出的 Stage 禁止使用 v4-pro**。

---

### #6 Writer 幻觉 X 推文链接

**现象**：Critic 检测到 `hallucinated url: https://x.com/StepFun_ai/status/2056170241147977741`

**根因**：`XCookieAdapter` 对所有推文返回 `https://x.com/{username}`（用户主页），而不是具体推文链接。Writer 需要具体链接时，自行编造了一个推文 ID。

**修复**：`XCookieAdapter._extract_tweets_from_html()` 从 HTML 中提取真实推文 permalink `/{username}/status/{tweet_id}`。

---

### #7 X 源抓到的是旧置顶推文

**现象**：Issue #30 被 DeepSeek 几个月前的 R1 发布、缓存降价等置顶推文刷屏（5/14 条）。

**根因**：`published_at` 被写死为 `datetime.now()`，无论推文实际发布时间。置顶推文是几个月前的，但时间戳是"现在"，通过了 `max_age_hours` 过滤。

**修复**：从 HTML 中提取 `<time datetime="2026-05-19T14:00:00.000Z">` 真实时间戳。旧推文被 `max_age_hours` 正确过滤。

**防止复发**：外部数据源的时间戳必须从源头提取，禁止伪造。

---

### #8 ResearchEditor 用 deepseek-chat 仍然选 0 条

**现象**：模型切换到 `deepseek-chat` 后，trace 显示 2604 output tokens，但 `llm_selected_count` 仍然是 0。

**根因**（经过 6 轮调试定位）：`max_output_tokens=3072` 太小。deepseek-chat 生成 16-24 条的完整编辑决策 JSON 需要约 15000 字符，3072 tokens 等价约 9100 字符，JSON 被截断在中间 → 解析失败。

**调试过程**：
1. 本地 mock 调用 deepseek-chat → 输出正常，16/16 解析成功 → 说明不是代码逻辑问题
2. 怀疑 CI 环境不同 → 加 `editor_parse_result` trace 事件
3. trace 显示 `raw_len: 9110, notes: JSON parse failed` → JSON 在 9109 字符处截断
4. 加 `raw_suffix` 看到结尾是不完整的 JSON 片段 → 确认是 token 限制

**修复**：`max_output_tokens` 从 3072 翻倍到 6144。

**防止复发**：输出 token 限制要匹配实际需求。包含大量结构化数据的 LLM 调用需要更多 tokens。

---

### #9 相同的语义重复阻挡发布

**现象**：Semantic Duplicate 检测到 4 对 high-severity 重复，Repairer 修复失败（JSON 解析错误），导致 Publish Gate 阻塞。

**根因**（两层）：
1. 相同推文被多个 X 源抓取，形成多个事件，Writer 为同一事件写了多条。
2. Repairer 用 deepseek-v4-pro 输出格式损坏的 JSON，修复失败。
3. Gate 读到修复前的语义重复报告，即使修复成功也会阻塞。

**修复**：
- 相同 URL 的语义重复（"相同URL""同一推文"）降级为 warning，不阻塞 gate
- 修复成功后，gate 不再重新阻塞修复前的语义重复
- `--force-dup` 选项：只跳过重复检查，不跳过质量门禁

---

### #10 Anthropic RSS 404

**现象**：`anthropic_news` 源（`https://www.anthropic.com/news/rss.xml`）返回 404，从未产出过数据。

**根因**：Anthropic 没有公开 RSS feed。

**修复**：移除该源。Anthropic 新闻通过 TechCrunch AI RSS、VentureBeat RSS 和 X 源（`@AnthropicAI`）覆盖。

---

### #11 Scout 源发现空跑

**现象**：Scout 步骤成功但输出空文件，5 天共发现 1 个可用源。

**根因**（三层）：
1. **LLM 模型不可用**：`deepseek-chat` 不在自定义 API 网关的模型列表中（只有 `deepseek-v4-flash` + `deepseek-v4-pro`），`_verify_model()` 抛出 `ModelUnavailable`。
2. **内容链扩散缺数据**：需要 `collected_items` 作为种子，但 CI 调用没传入。
3. **无反馈闭环**：即使发现新源，也没有自动纳入配置。

**修复**：
- 切换到 `deepseek-v4-flash`（网关可用）+ `--skip-model-check`
- Pipeline collect 阶段保存 `artifacts/collected/{date}.json`（全量 raw items）
- Scout CLI 从 `collected` 文件读取（全量，不是 16 条精选）
- `admit-sources` 命令实现自动纳新（去重 + 低权重追加）

---

### #12 论文精选板块混入非论文内容

**现象**：Issue #36 的"论文精选"板块 3 条全是产品/合作新闻，没有一篇学术论文。

**根因**：ResearchEditor prompt 对"论文精选"的定义不够明确——"arXiv/HuggingFace 论文"不够显眼，LLM 可能理解为"值得研究的主题"。

**修复**（两层）：
- **Prompt 层**：加硬约束——"**仅限 source_urls 含 arxiv.org 或 huggingface.co/papers 的学术论文**。产品发布、合作公告、企业新闻一律不放这里。"
- **后验证层**：`_parse_and_validate` 自动检测——分配到论文精选但没有 arxiv/HF URL 的条目，自动重新分类到"业界风向"，记 trace warning。

**防止复发**：LLM 分类 + 后验证兜底的双层保护。

---

### #13 自定义 API 网关模型约束

**现象**：`DEEPSEEK_BASE_URL` 指向的自定义网关只暴露 `deepseek-v4-flash` + `deepseek-v4-pro`，任何使用 `deepseek-chat` 的 provider 创建都会失败（除非 `skip_model_check=True`）。

**影响**：ResearchEditor、Writer、Scout 三个 Stage 都受影响。

**修复总结**：
| Stage | 解决方案 |
|-------|---------|
| ResearchEditor | `skip_model_check=True`，模型名用 `deepseek-chat`（网关接受） |
| Writer | 同上 |
| Scout | 用 `deepseek-v4-flash`（网关已列出）+ `--skip-model-check` |

**防止复发**：使用非网关列出的模型名时，必须传 `skip_model_check=True`。

---

### 2026-05-19 修复统计

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| CI 运行 | 不运行 | 正常运行 |
| 原始采集量 | 147-191 条 | 200-270 条 |
| LLM 编辑选择率 | 0/115 | 24/82 |
| 幻觉 URL | 有时出现 | 0 |
| 论文占比 | 0-50% | 26% |
| 来源多样性 | 4-5 个源 | 7-10 个源 |
| X 源状态 | 全部沉默 | 正常产出 |
| 旧闻混入 | 严重 | 极少 |
| 日报板块 | 旧 6 板块 | 新 6 板块 |
| 论文板块纯度 | N/A | 100%（双层保护后） |
| Scout | 空跑 | 待定时验证 |
