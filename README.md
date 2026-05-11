# AI 日报

全自动 AI 科技日报生成系统。每日 12:00 从 30+ 信息源采集全球 AI 资讯，经多 Agent 协作流水线处理后，发布到 GitHub Issues 并推送 PDF 到订阅者邮箱。

## 系统架构

```
                             AI 日报自动化流水线
 ┌──────────────────────────────────────────────────────────────────────┐
 │                                                                      │
 │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐          │
 │  │ Collector │ → │ Curator  │ → │  Writer  │ → │  Critic  │          │
 │  │ 30+ RSS   │   │ 去重/排序 │   │ DeepSeek │   │ 幻觉检测 │          │
 │  │ X/API/Web │   │ 相关性   │   │ 深度写稿 │   │ 语义去重 │          │
 │  └──────────┘   └──────────┘   └──────────┘   └─────┬────┘          │
 │                                                     │               │
 │                              ┌──────────┐           │               │
 │                              │ Repairer │ ←─────────┘               │
 │                              │ 自动修复  │                           │
 │                              └──────────┘                           │
 │                                     │                                │
 │                    ┌────────────────┴────────────────┐               │
 │                    ↓                                  ↓               │
 │  ┌──────────┐   ┌──────────┐   ┌──────────┐                        │
 │  │ Publisher│   │   PDF    │ → │  Email   │                        │
 │  │ Issue #N │   │ Generator│   │ SMTP QQ  │                        │
 │  └──────────┘   └──────────┘   └──────────┘                        │
 └──────────────────────────────────────────────────────────────────────┘

 监控层:
 ┌──────────┐   ┌──────────┐   ┌──────────┐
 │  Web 看板 │   │ 健康检查 │   │  告警    │
 │ FastAPI   │   │ RSS 新鲜度│   │ 邮件通知 │
 │ 归档/统计 │   │ 流水线状态│   │ 连续失败 │
 └──────────┘   └──────────┘   └──────────┘
```

## 技术栈

| 层级 | 技术 |
|------|------|
| 语言 | Python 3.12+ |
| LLM | DeepSeek V4 Pro |
| 数据采集 | feedparser, X API v2, Playwright, custom scrapers |
| 流水线 | 7-Agent 协作架构 |
| Web 看板 | FastAPI + Jinja2 |
| PDF | Playwright headless Chromium |
| 部署 | Docker Compose, GitHub Actions |
| 测试 | pytest (138 tests, offline mock) |

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env 填入 DEEPSEEK_API_KEY 等
python -m agent.cli run --provider deepseek
python -m agent.cli publish-issue --run-id $(date +%Y-%m-%d) --confirm
python -m uvicorn agent.web.app:app --host 0.0.0.0 --port 8080
```

## Web 看板

```
http://localhost:8080           — 首页
http://localhost:8080/archive   — 归档
http://localhost:8080/stats     — 统计
http://localhost:8080/monitor   — 监控
```

## REST API

```
GET  /api/reports
GET  /api/reports/2026-05-10
GET  /api/stats
GET  /api/monitor
POST /api/subscribe?email=xxx
POST /api/unsubscribe?email=xxx
```

## 自动化

| Workflow | 触发 | 功能 |
|----------|------|------|
| `daily_run.yml` | 每天 12:00 | 采集→写稿→发布→邮件 |
| `subscribe.yml` | Issue 评论 | 邮箱验证→自动订阅/退订 |
| `health_check.yml` | 每周一 | 信源扫描→告警 |

## License

MIT
