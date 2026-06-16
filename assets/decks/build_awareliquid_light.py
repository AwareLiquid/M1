# -*- coding: utf-8 -*-
"""Build the AwareLiquid "Investor Deck Light" as a styled HTML, ready for
Chrome headless --print-to-pdf. Visual style mirrors the prior beamer deck
(serif CJK body, "Awareness" wordmark, 3-box footline, logo top-right), but the
CONTENT is the honest, graded (Built / Early-signal / Target) 5-minute version.
"""
import base64
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
LOGO = base64.b64encode((ROOT / "assets/figures/logo.png").read_bytes()).decode()

# ---- palette (sober, investor-grade) ----
INK = "#15202b"        # near-black body ink
MUTE = "#5b6b7a"       # muted gray
ACCENT = "#1f6feb"     # liquid blue
LINE = "#d7dee6"       # hairline
BG = "#ffffff"
BUILT = "#15803d"      # green
SIGNAL = "#b45309"     # amber
TARGET = "#6d28d9"     # violet

SLIDES = []


def slide(html):
    SLIDES.append(html)


# 1. Title -----------------------------------------------------------------
slide(f"""
<section class="slide title">
  <div class="brand-mark">Awareness</div>
  <h1>AwareLiquid</h1>
  <div class="sub">全栈自研的液态神经网络类脑架构<br>
  <span class="sub2">Investor Deck · Light · 五分钟精华版</span></div>
  <div class="title-meta">
    <div><span class="k">GitHub</span> github.com/everest-an/M1</div>
    <div><span class="k">HuggingFace</span> huggingface.co/EverestAn/MT-LNN</div>
    <div><span class="k">Live demo</span> awareliquid.ai（部署中）</div>
  </div>
  <div class="honesty">诚实分级原则：每条结论标注为 <b class="bd">Built</b> / <b class="sg">Early&nbsp;signal</b> / <b class="tg">Target</b>。<br>
  无法通过技术尽调的数字，不写进本 deck。</div>
</section>
""")

# 2. One line + product lines ---------------------------------------------
slide(f"""
<section class="slide">
  <h2>一句话</h2>
  <p class="lead">AwareLiquid 是国内首个<b>全栈自研的液态神经网络类脑架构</b>。
  我们不走参数堆叠的老路，而是从<b>连续时间动力学</b>这一根上重做底座——
  押注的是 Transformer 之外的<b>第二条技术路线</b>。</p>
  <div class="cards">
    <div class="card">
      <div class="card-tag">产品线 O1</div>
      <div class="card-title">液态时序基座</div>
      <div class="card-body">Liquid temporal foundation。原生处理连续时序与长流数据的底座模型。</div>
    </div>
    <div class="card">
      <div class="card-tag">产品线 M1</div>
      <div class="card-title">自主认知智能体</div>
      <div class="card-body">感知编码 → 空间记忆 → 目标调制 → 想象推演 → 安全输出 的闭环。是主动体，不是被动应答工具。</div>
    </div>
  </div>
</section>
""")

# 3. Pain point / why now --------------------------------------------------
slide(f"""
<section class="slide">
  <h2>痛点 / 为什么是现在</h2>
  <p class="lead">Transformer 大模型在三类场景天然吃力——而这些恰是工业与高合规市场的刚需。</p>
  <ol class="big-list">
    <li><b>端侧落地难</b>　参数大、算力高，难在边缘设备低功耗长驻。</li>
    <li><b>时序场景弱</b>　长序列幻觉累积、连续流处理效率低、分布外（OOD）泛化差。</li>
    <li><b>自主性缺失</b>　只能被动应答，无法自主发现问题、设目标、闭环决策；高可靠场景缺原生安全兜底。</li>
  </ol>
  <div class="footnote">目标场景：工业控制 · 端侧机器人 · 高合规金融/政企 · 边缘监测。</div>
</section>
""")

# 4. Differentiation -------------------------------------------------------
slide(f"""
<section class="slide">
  <h2>差异化：架构根源，不是调参</h2>
  <div class="rows">
    <div class="row">
      <div class="row-h">液态时间常数神经元 + 层级预测编码</div>
      <div class="row-b">原生适配连续时序数据，自带误差自校正机制——这是<b>架构层面</b>的根源差异，不是在 Transformer 上做的优化。</div>
    </div>
    <div class="row">
      <div class="row-h">双产品矩阵</div>
      <div class="row-b">O1 基座 + M1 主动智能体。M1 是主动体，能自主设目标、闭环决策，而非被动应答。</div>
    </div>
    <div class="row">
      <div class="row-h">全栈自研、完全开源可复现</div>
      <div class="row-b">代码、模型、测试全部公开，可被任何第三方独立验证。</div>
    </div>
  </div>
</section>
""")

# 5. Evidence, graded (the core slide) ------------------------------------
slide(f"""
<section class="slide">
  <h2>进展与证据 <span class="h2-note">— 诚实分级，这页是尽调友好的核心</span></h2>
  <div class="grade">
    <div class="grade-head bd">✅ Built — 已构建并可复现</div>
    <ul>
      <li><b>125M 类脑基座端到端跑通</b>：WikiText-103 从零预训练，当前 held-out val PPL ≈ 136，仍在迭代收敛。</li>
      <li><b>M1 自主认知闭环</b>：感知-记忆-目标-想象-安全输出全链路已实现。</li>
      <li><b>1083 项行为契约测试</b>（100 个测试文件）：输出可审计、可复现。</li>
    </ul>
  </div>
  <div class="grade">
    <div class="grade-head sg">🔬 Early signal — 小规模、带误差棒的方向性证据</div>
    <ul>
      <li>受控抗遗忘探针上，液态预测编码核心遗忘率<b>显著低于 GRU 与 Transformer 基线</b>（5 seed，带误差棒）。<span class="scope">范围：~1.5–2 万参数级合成时序任务；尚未在语言模型规模复现——这正是下一步。</span></li>
      <li>生成式回放（无需存原始数据）在同探针上大幅降低遗忘。<span class="scope">范围：该效应在 RNN 基线上同样出现，应理解为「回放方法有效」，架构独占性仍待证明。</span></li>
    </ul>
  </div>
  <div class="grade">
    <div class="grade-head tg">🎯 Target — 本轮要证明的事（资金用途）</div>
    <ul>
      <li>把时序与抗遗忘优势从小规模<b>迁移到语言模型规模</b>，做出对得过同尺寸 baseline 的带误差棒对照。</li>
      <li>跨域灾难性遗忘对照（真正的领域切换，而非同域切片）。</li>
      <li>端侧路径：用蒸馏让小模型在<b>特定窄任务</b>上对标更大模型——真实可达，而非通用基准正面对打。</li>
    </ul>
  </div>
</section>
""")

# 6. Go-to-market + milestones --------------------------------------------
slide(f"""
<section class="slide">
  <h2>商业切入 + 里程碑</h2>
  <p class="lead"><b>策略</b>：先做<b>窄域 + 端侧</b>，用具体场景的可靠性 / 时序 / 功耗优势打开局面；
  不在通用基准上正面与大模型竞争。</p>
  <div class="cols">
    <div class="col">
      <div class="col-h">里程碑</div>
      <ol class="ms">
        <li>LM 规模的抗遗忘 / 长程对照（带误差棒 vs 同尺寸 baseline）。</li>
        <li>1–2 个标杆场景 POC（工业控制或边缘监测优先）。</li>
        <li>上线公开 demo <b>awareliquid.ai</b>（推理服务 + TLS 部署栈已完成，待基座 checkpoint 接入）。</li>
      </ol>
    </div>
    <div class="col">
      <div class="col-h">资金用途</div>
      <ol class="ms">
        <li>规模化验证算力。</li>
        <li>标杆场景落地工程。</li>
      </ol>
    </div>
  </div>
</section>
""")

# 7. Investment logic + verifiable assets ---------------------------------
slide(f"""
<section class="slide closing">
  <h2>投资逻辑</h2>
  <p class="lead big">这是一次<b>架构路线的早期押注</b>：底层动力学差异化已成型、全栈工程已可复现、
  小规模方向性信号已出现。我们要用本轮资金，把这些信号在真实规模上
  <b>坐实成带误差棒的、独立可验证的优势</b>——而不是急于声称尚未验证的结果。</p>
  <div class="assets">
    <div class="assets-h">可独立验证的资产</div>
    <div class="assets-row"><span class="k">代码与实验报告</span> github.com/everest-an/M1</div>
    <div class="assets-row"><span class="k">模型页</span> huggingface.co/EverestAn/MT-LNN</div>
    <div class="assets-row"><span class="k">Live demo（部署中）</span> awareliquid.ai</div>
  </div>
  <div class="footnote">本 deck 的每一项数据均可在开源仓库中追溯到对应实验脚本与报告。</div>
</section>
""")

N = len(SLIDES)
footed = []
for i, s in enumerate(SLIDES, 1):
    foot = f"""
  <div class="footline">
    <div class="fl fl-l">Awareness</div>
    <div class="fl fl-c">{i} / {N}</div>
    <div class="fl fl-r">Confidential &amp; Proprietary</div>
  </div>
  <img class="logo" src="data:image/png;base64,{LOGO}" alt="logo">
"""
    s = s.replace("</section>", foot + "</section>")
    footed.append(s)

CSS = f"""
@page {{ size: 33.867cm 19.05cm; margin: 0; }}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body {{ background: {BG}; color: {INK};
  font-family: "SimSun","Songti SC","Noto Serif CJK SC",serif; }}
.slide {{
  position: relative; width: 33.867cm; height: 19.05cm;
  padding: 1.5cm 2.0cm 1.6cm 2.0cm; overflow: hidden;
  page-break-after: always; display: flex; flex-direction: column;
}}
.slide:last-child {{ page-break-after: auto; }}
h2 {{ font-family: "Microsoft YaHei","Segoe UI",sans-serif; font-weight: 700;
  font-size: 30px; color: {INK}; letter-spacing: .3px;
  padding-bottom: .35cm; margin-bottom: .5cm;
  border-bottom: 2px solid {ACCENT}; }}
.h2-note {{ font-size: 16px; font-weight: 400; color: {MUTE}; }}
.lead {{ font-size: 20px; line-height: 1.7; color: {INK}; margin-bottom: .55cm; }}
.lead.big {{ font-size: 22px; line-height: 1.8; }}
.lead b, .row-b b, li b, .card-body b {{ color: {ACCENT}; font-weight: 700; }}
.footnote {{ margin-top: auto; font-size: 14px; color: {MUTE};
  border-top: 1px solid {LINE}; padding-top: .3cm; }}

/* footline + logo */
.footline {{ position: absolute; left: 0; right: 0; bottom: .55cm;
  display: flex; font-family: "Microsoft YaHei","Segoe UI",sans-serif;
  font-size: 12px; color: {MUTE}; }}
.fl {{ flex: 1; }}
.fl-l {{ padding-left: 2.0cm; letter-spacing: 1px; }}
.fl-c {{ text-align: center; }}
.fl-r {{ text-align: right; padding-right: 2.0cm; letter-spacing: .5px; }}
.logo {{ position: absolute; top: .5cm; right: .55cm; height: 1.05cm; opacity: .92; }}

/* title slide */
.title {{ justify-content: center; }}
.brand-mark {{ font-family: "Aeonik","Microsoft YaHei","Segoe UI",sans-serif;
  font-size: 26px; letter-spacing: 3px; color: {MUTE}; margin-bottom: .5cm; }}
.title h1 {{ font-family: "Aeonik","Microsoft YaHei","Segoe UI",sans-serif;
  font-size: 84px; font-weight: 800; letter-spacing: 1px; color: {INK}; }}
.title .sub {{ font-size: 24px; line-height: 1.6; margin-top: .35cm; color: {INK}; }}
.title .sub2 {{ font-size: 17px; color: {MUTE}; }}
.title-meta {{ margin-top: 1.0cm; font-size: 16px; line-height: 1.9;
  font-family: "Microsoft YaHei","Segoe UI",sans-serif; color: {INK}; }}
.title-meta .k {{ display: inline-block; width: 3.6cm; color: {ACCENT}; font-weight: 700; }}
.honesty {{ margin-top: 1.0cm; font-size: 15px; line-height: 1.7; color: {MUTE};
  border-left: 3px solid {ACCENT}; padding-left: .5cm; }}
.honesty .bd {{ color: {BUILT}; }} .honesty .sg {{ color: {SIGNAL}; }} .honesty .tg {{ color: {TARGET}; }}

/* cards */
.cards {{ display: flex; gap: .8cm; margin-top: .3cm; }}
.card {{ flex: 1; border: 1px solid {LINE}; border-top: 4px solid {ACCENT};
  border-radius: 6px; padding: .6cm .7cm; background: #fbfcfe; }}
.card-tag {{ font-family: "Microsoft YaHei",sans-serif; font-size: 14px;
  color: {ACCENT}; font-weight: 700; letter-spacing: .5px; }}
.card-title {{ font-size: 24px; font-weight: 700; margin: .15cm 0 .25cm; }}
.card-body {{ font-size: 17px; line-height: 1.65; color: {INK}; }}

/* big numbered list */
.big-list {{ list-style: none; counter-reset: bl; }}
.big-list li {{ counter-increment: bl; position: relative; font-size: 20px;
  line-height: 1.6; padding: .28cm 0 .28cm 1.25cm; border-bottom: 1px solid {LINE}; }}
.big-list li:before {{ content: counter(bl); position: absolute; left: 0; top: .28cm;
  width: .85cm; height: .85cm; background: {ACCENT}; color: #fff; border-radius: 50%;
  font-family: "Microsoft YaHei",sans-serif; font-size: 15px; font-weight: 700;
  display: flex; align-items: center; justify-content: center; }}

/* differentiation rows */
.rows {{ display: flex; flex-direction: column; gap: .5cm; }}
.row {{ border: 1px solid {LINE}; border-left: 4px solid {ACCENT}; border-radius: 5px;
  padding: .45cm .7cm; }}
.row-h {{ font-size: 21px; font-weight: 700; margin-bottom: .12cm; }}
.row-b {{ font-size: 17px; line-height: 1.6; color: {INK}; }}

/* graded evidence */
.grade {{ margin-bottom: .35cm; }}
.grade-head {{ font-family: "Microsoft YaHei",sans-serif; font-size: 18px;
  font-weight: 700; padding: .12cm 0 .1cm; }}
.grade-head.bd {{ color: {BUILT}; }} .grade-head.sg {{ color: {SIGNAL}; }} .grade-head.tg {{ color: {TARGET}; }}
.grade ul {{ list-style: none; }}
.grade li {{ position: relative; font-size: 15.5px; line-height: 1.5;
  padding: .05cm 0 .05cm .5cm; }}
.grade li:before {{ content: ""; position: absolute; left: 0; top: .28cm;
  width: 5px; height: 5px; background: {MUTE}; border-radius: 50%; }}
.scope {{ display: block; font-size: 13px; color: {MUTE}; font-style: italic; }}

/* columns */
.cols {{ display: flex; gap: 1.0cm; }}
.col {{ flex: 1; }}
.col-h {{ font-family: "Microsoft YaHei",sans-serif; font-size: 16px; font-weight: 700;
  color: {ACCENT}; letter-spacing: .5px; margin-bottom: .25cm;
  border-bottom: 1px solid {LINE}; padding-bottom: .15cm; }}
.ms {{ padding-left: .6cm; }}
.ms li {{ font-size: 18px; line-height: 1.55; margin-bottom: .25cm; }}

/* closing */
.assets {{ margin-top: .4cm; border: 1px solid {LINE}; border-radius: 6px;
  padding: .55cm .8cm; background: #fbfcfe; }}
.assets-h {{ font-family: "Microsoft YaHei",sans-serif; font-size: 16px; font-weight: 700;
  color: {ACCENT}; margin-bottom: .25cm; }}
.assets-row {{ font-family: "Microsoft YaHei",sans-serif; font-size: 17px; line-height: 1.9; }}
.assets-row .k {{ display: inline-block; width: 5.0cm; color: {MUTE}; }}
"""

HTML = f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>AwareLiquid — Investor Deck Light</title>
<style>{CSS}</style></head>
<body>
{''.join(footed)}
</body></html>"""

out = ROOT / "assets/decks/AwareLiquid_Investor_Deck_Light.html"
out.write_text(HTML, encoding="utf-8")
print("wrote", out, len(HTML), "bytes,", N, "slides")
