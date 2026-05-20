# AI Frontier & Market Intelligence Agent

面向 AI 从业者、研究者、开发者、产品/创业者和市场观察者的多源情报 Agent。

跟踪模型能力、API 价格、研究前沿、产品工作流、产业战略、资本市场、AI infra、硅谷 insider 信号和国内大模型生态。通过 Agent Harness 完成采集、筛选、证据约束生成、质量审查、语义去重、自动修复和发布门控。

## 这是一个独立项目

本项目是一个独立的 AI Frontier & Market Intelligence Agent，不是一个上游插件或寄生仓库。
- juya-ai-daily 只作为早期 AI 早报版式参考，不是运行依赖
- 不接入原 juya-ai-daily 的 generate_readme.yml 等下游工作流
- 不验证原下游 README/RSS/Pages

系统的核心价值在于 Source Coverage + Evidence-Grounded Synthesis + Quality Gate + Repair Loop + Trace/Replay/Eval，而不是单次 LLM 总结。内容版式参考高质量 AI 早报，GitHub Issue / Markdown / JSON 是当前阶段的输出形式。

## 它要回答的问题

```
1. 今天模型能力边界有没有变化？
2. 哪些模型/API/订阅价格发生变化？哪个更划算？
3. 哪些论文或 benchmark 暗示研究方向变化？
4. 哪些产品/工具真正影响 AI 工作流？
5. 哪些大厂/创业公司战略变化值得关注？
6. 哪些 AI infra、GPU、云、推理框架、数据中心信号重要？
7. 哪些资本市场/财报/capex 信号影响 AI 产业链？
8. 硅谷 insider、VC、builder、社区在讨论什么？
9. 国内大模型厂商和开发者生态发生了什么变化？
10. 这些变化对 researcher / developer / founder / product / investor 分别意味着什么？
```

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│              AI Frontier & Market Intelligence Agent            │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ Collector │→ │ Curator  │→ │  Writer  │→ │  Critic  │        │
│  │ 56+ 源    │  │ 去重/排序 │  │ DeepSeek │  │ 幻觉检测 │        │
│  │ content-  │  │ half-life│  │ evidence- │  │ 语义去重 │        │
│  │ type aware│  │ scoring  │  │ grounded  │  │          │        │
│  └──────────┘  └──────────┘  └──────────┘  └────┬─────┘        │
│                                                  │              │
│                           ┌──────────┐           │              │
│                           │ Repairer │←──────────┘              │
│                           │ 自动修复  │                           │
│                           └──────────┘                           │
│                                │                                 │
│              ┌─────────────────┴──────────────┐                  │
│              ↓                                ↓                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                      │
│  │ Publisher│  │   PDF    │→ │  Email   │                      │
│  │ Issue #N │  │ Generator│  │ SMTP QQ  │                      │
│  └──────────┘  └──────────┘  └──────────┘                      │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                      │
│  │ Web 看板  │  │ 健康检查 │  │  告警    │                      │
│  │ FastAPI   │  │ 源新鲜度 │  │ 邮件通知 │                      │
│  └──────────┘  └──────────┘  └──────────┘                      │
└─────────────────────────────────────────────────────────────────┘
```

## Content Type 分类体系

| 分类 | 说明 | weight | half-life |
|------|------|--------|-----------|
| official_release | 公司正式发布 | 1.35 | 48h |
| official_docs | API/模型文档 | 1.35 | 168h |
| pricing_page | 官方定价页 | 1.35 | 168h |
| benchmark_tracker | 能力榜/性能榜 | 1.20 | 72h |
| research_paper | 学术论文 | 1.05 | 120h |
| china_model_official | 国内模型厂商发布 | 1.25 | 48h |
| china_model_docs | 国内模型API文档 | 1.30 | 168h |
| china_model_pricing | 国内模型定价 | 1.35 | 168h |
| insider_media | 有编辑流程的insider报道 | 1.05 | 72h |
| infra_signal | 推理框架/GPU/云 | 1.10 | 120h |
| financial_report | 财报/IR/capex | 1.30 | 120h |
| founder_signal | 创始人/高管观点 | 1.00 | 36h |
| researcher_signal | 研究者观点 | 0.95 | 72h |
| builder_signal | builder实践信号 | 0.90 | 72h |
| tech_media | 科技媒体 | 0.85 | 36h |
| vc_signal | VC/投资人信号 | 0.80 | 36h |
| community_signal | 社区信号 | 0.75 | 24h |

## 质量保证

- **Publish Gate**: critic verdict + eval issues + semantic duplicates + repair failure
- **Evidence Grounding**: 每条 item 标注 source_url, evidence_type, confidence  
- **Content-Type-Aware Scoring**: 真正的 half-life 衰减公式 `exp(-ln(2)*age/hl)`
- **Research Quota**: 论文保底 5 篇，backfill 机制
- **Token Budget Policy**: 硬限制防止源扩展后 token 爆炸

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env 填入 DEEPSEEK_API_KEY 等
python -m agent.cli run --provider deepseek
```

## 技术栈

Python 3.12+ · DeepSeek · Playwright · FastAPI · Pydantic v2 · Docker · GitHub Actions

## License

MIT
