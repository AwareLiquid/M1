# -*- coding: utf-8 -*-
"""Build the AwareLiquid "Investor Deck Light" as a styled HTML, ready for
Chrome headless --print-to-pdf.

Design: a nature / organic ("liquid", brain-inspired) aesthetic -- botanical
green + warm-sand palette, soft water-like gradients, larger type. Visual
lineage keeps the prior deck's "Awareness" wordmark, 3-box footline, and
logo top-right. CONTENT is the honest, graded (Built / Early-signal / Target)
5-minute version, now with the strongest NARRATIVE beats ported from the old
deck (memory crisis / KV-cache, microtubule bio-inspiration) -- but WITHOUT
the old deck's fabricated benchmark numbers.
"""
import base64
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
FIG = ROOT / "assets/figures"


def b64(path):
    return base64.b64encode(pathlib.Path(path).read_bytes()).decode()


LOGO = b64(FIG / "logo.png")
IMG_COST = b64(FIG / "cost_explosion.png")
IMG_MICRO = b64(FIG / "fig_microtubules.png")
IMG_ARCH = b64(FIG / "fig_architecture.png")

# ---- nature / organic palette ----
BG = "#faf8f2"          # warm paper
PANEL = "#f2f0e6"       # soft sand panel
INK = "#1c2b25"         # deep forest ink
MUTE = "#5e6b62"        # sage gray
ACCENT = "#1f7a5a"      # liquid leaf-green
ACCENT_D = "#155c43"    # deep green
LINE = "#d8d6c8"        # natural hairline
BUILT = "#1f7a5a"       # green (Built)
SIGNAL = "#b3781f"      # amber (Early signal)
TARGET = "#6a4ea8"      # muted violet (Target)

SLIDES = []


def slide(html):
    SLIDES.append(html)


# 1. Title -----------------------------------------------------------------
slide(f"""
<section class="slide title">
  <div class="leaf-bg"></div>
  <div class="brand-mark">Awareness</div>
  <h1>AwareLiquid</h1>
  <div class="sub">全栈自研的<b>液态神经网络</b>类脑架构</div>
  <div class="sub2">Investor Deck · Light · 五分钟精华版</div>
  <div class="title-meta">
    <div><span class="k">GitHub</span> github.com/everest-an/M1</div>
    <div><span class="k">HuggingFace</span> huggingface.co/EverestAn/MT-LNN</div>
    <div><span class="k">Live demo</span> awareliquid.ai（部署中）</div>
  </div>
  <div class="honesty">诚实分级：每条结论标注
    <b class="bd">Built</b> · <b class="sg">Early&nbsp;signal</b> · <b class="tg">Target</b>；
    无法通过技术尽调的数字，不写进本 deck。</div>
</section>
""")

# 2. Memory crisis / why now (ported narrative) ---------------------------
slide(f"""
<section class="slide">
  <h2>AI 的内存危机 — 为什么是现在</h2>
  <div class="split">
    <div class="split-l">
      <ul class="big-list">
        <li><b>强迫式记录引擎</b>　Transformer 预测下一个词，必须把过去所有词放进显存（KV Cache）。</li>
        <li><b>复杂度灾难</b>　文本越长，内存 <i>O(N)</i>、算力飙到 <i>O(N²)</i>——成本指数级爆炸。</li>
        <li><b>端侧不可能</b>　十几 GB 内存的手机 / 边缘设备，物理上塞不下长上下文。</li>
      </ul>
      <div class="callout">这正是工业控制、端侧机器人、高合规政企的刚需缺口。</div>
    </div>
    <div class="split-r">
      <img class="fig" src="data:image/png;base64,{IMG_COST}" alt="cost">
      <div class="fig-cap">概念示意：复杂度量级对比（asymptotic），非实测基准。</div>
    </div>
  </div>
</section>
""")

# 3. Bio-inspiration / microtubules (ported narrative) --------------------
slide(f"""
<section class="slide">
  <h2>生物学启示：从人脑到微管</h2>
  <div class="split">
    <div class="split-l">
      <ul class="big-list">
        <li><b>大脑不记录每个像素</b>　认知靠「工作记忆」滞存核心线索，靠「选择性遗忘」抛弃噪音。</li>
        <li><b>微管：神经计算的底层基座</b>　神经元内部的细胞骨架网络，以极高频率动态过滤、压缩信息。</li>
        <li><b>从生物到 LNN</b>　我们摒弃强制缓存历史，用类微管的<b>连续微分方程</b>，让信息随时间流动、留存与遗忘。</li>
      </ul>
    </div>
    <div class="split-r">
      <img class="fig" src="data:image/png;base64,{IMG_MICRO}" alt="microtubules">
      <div class="fig-cap">受微管动态启发的「液态状态流」灵感链。</div>
    </div>
  </div>
</section>
""")

# 4. One line + product lines ---------------------------------------------
slide(f"""
<section class="slide">
  <h2>一句话 + 两条产品线</h2>
  <p class="lead">AwareLiquid 是国内首个<b>全栈自研的液态神经网络类脑架构</b>。
  不走参数堆叠的老路，而是从<b>连续时间动力学</b>这一根上重做底座——
  押注 Transformer 之外的<b>第二条技术路线</b>。</p>
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

# 5. Differentiation -------------------------------------------------------
slide(f"""
<section class="slide">
  <h2>差异化：架构根源，不是调参</h2>
  <div class="split">
    <div class="split-l">
      <div class="rows">
        <div class="row">
          <div class="row-h">液态时间常数神经元 + 层级预测编码</div>
          <div class="row-b">原生适配连续时序，自带误差自校正——<b>架构层面</b>的根源差异，不是在 Transformer 上调优。</div>
        </div>
        <div class="row">
          <div class="row-h">双产品矩阵：O1 基座 + M1 主动体</div>
          <div class="row-b">M1 能自主设目标、闭环决策，而非被动应答。</div>
        </div>
        <div class="row">
          <div class="row-h">全栈自研、完全开源可复现</div>
          <div class="row-b">代码、模型、测试全部公开，可被第三方独立验证。</div>
        </div>
      </div>
    </div>
    <div class="split-r">
      <img class="fig" src="data:image/png;base64,{IMG_ARCH}" alt="architecture">
      <div class="fig-cap">MT-LNN 架构示意。</div>
    </div>
  </div>
</section>
""")

# 6. Evidence, graded (core slide) ----------------------------------------
slide(f"""
<section class="slide">
  <h2>进展与证据 <span class="h2-note">— 诚实分级，尽调友好</span></h2>
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

# 7. Go-to-market + milestones --------------------------------------------
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

# 8. Investment logic + verifiable assets ---------------------------------
slide(f"""
<section class="slide closing">
  <div class="leaf-bg"></div>
  <h2>投资逻辑</h2>
  <p class="lead big">这是一次<b>架构路线的早期押注</b>：底层动力学差异化已成型、全栈工程已可复现、
  小规模方向性信号已出现。本轮资金，把这些信号在真实规模上
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
    footed.append(s.replace("</section>", foot + "</section>"))

CSS = f"""
@page {{ size: 33.867cm 19.05cm; margin: 0; }}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body {{ background: {BG}; color: {INK};
  font-family: "SimSun","Songti SC","Noto Serif CJK SC",serif; }}
.slide {{
  position: relative; width: 33.867cm; height: 19.05cm;
  padding: 1.5cm 2.0cm 1.7cm 2.0cm; overflow: hidden;
  page-break-after: always; display: flex; flex-direction: column;
  background:
    radial-gradient(120% 80% at 100% 0%, rgba(31,122,90,.06), transparent 55%),
    radial-gradient(90% 70% at 0% 100%, rgba(31,122,90,.05), transparent 55%),
    {BG};
}}
.slide:last-child {{ page-break-after: auto; }}

h2 {{ font-family: "Microsoft YaHei","Segoe UI",sans-serif; font-weight: 700;
  font-size: 38px; color: {INK}; letter-spacing: .3px;
  padding-bottom: .35cm; margin-bottom: .55cm; position: relative; }}
h2:after {{ content: ""; position: absolute; left: 0; bottom: 0;
  width: 3.2cm; height: 4px; background: {ACCENT}; border-radius: 3px; }}
.h2-note {{ font-size: 20px; font-weight: 400; color: {MUTE}; }}

.lead {{ font-size: 25px; line-height: 1.75; color: {INK}; margin-bottom: .55cm; }}
.lead.big {{ font-size: 27px; line-height: 1.85; }}
.lead b, .row-b b, li b, .card-body b, .callout b {{ color: {ACCENT_D}; font-weight: 700; }}
.footnote {{ margin-top: auto; font-size: 17px; color: {MUTE};
  border-top: 1px solid {LINE}; padding-top: .3cm; }}

/* footline + logo */
.footline {{ position: absolute; left: 0; right: 0; bottom: .55cm;
  display: flex; font-family: "Microsoft YaHei","Segoe UI",sans-serif;
  font-size: 13px; color: {MUTE}; }}
.fl {{ flex: 1; }}
.fl-l {{ padding-left: 2.0cm; letter-spacing: 1px; }}
.fl-c {{ text-align: center; }}
.fl-r {{ text-align: right; padding-right: 2.0cm; letter-spacing: .5px; }}
.logo {{ position: absolute; top: .55cm; right: .6cm; height: 1.15cm; opacity: .9; }}

/* split layout (text + figure) */
.split {{ display: flex; gap: 1.0cm; flex: 1; align-items: center; }}
.split-l {{ flex: 1.15; }}
.split-r {{ flex: .95; text-align: center; }}
.fig {{ max-width: 100%; max-height: 11.5cm; border-radius: 8px;
  background: #fff; padding: .35cm; border: 1px solid {LINE};
  box-shadow: 0 6px 22px rgba(21,92,67,.10); }}
.fig-cap {{ font-size: 15px; color: {MUTE}; font-style: italic; margin-top: .25cm; }}
.callout {{ margin-top: .5cm; font-size: 20px; line-height: 1.6; color: {ACCENT_D};
  background: rgba(31,122,90,.08); border-left: 4px solid {ACCENT};
  border-radius: 0 6px 6px 0; padding: .4cm .6cm; }}

/* title slide */
.title {{ justify-content: flex-start; padding-top: 2.6cm; }}
.leaf-bg {{ position: absolute; inset: 0; z-index: 0;
  background:
    radial-gradient(60% 90% at 88% 18%, rgba(31,122,90,.16), transparent 60%),
    radial-gradient(70% 70% at 12% 92%, rgba(21,92,67,.12), transparent 60%); }}
.title > *:not(.footline):not(.logo),
.closing > *:not(.footline):not(.logo) {{ position: relative; z-index: 1; }}
.brand-mark {{ font-family: "Aeonik","Microsoft YaHei","Segoe UI",sans-serif;
  font-size: 30px; letter-spacing: 4px; color: {ACCENT}; margin-bottom: .45cm; }}
.title h1 {{ font-family: "Aeonik","Microsoft YaHei","Segoe UI",sans-serif;
  font-size: 84px; font-weight: 800; letter-spacing: 1px; color: {INK};
  line-height: 1.0; }}
.title .sub {{ font-size: 30px; line-height: 1.5; margin-top: .4cm; color: {INK}; }}
.title .sub2 {{ font-size: 21px; color: {MUTE}; margin-top: .15cm; }}
.title-meta {{ margin-top: .8cm; font-size: 19px; line-height: 1.9;
  font-family: "Microsoft YaHei","Segoe UI",sans-serif; color: {INK}; }}
.title-meta .k {{ display: inline-block; width: 4.0cm; color: {ACCENT}; font-weight: 700; }}
.honesty {{ margin-top: .7cm; font-size: 16px; line-height: 1.55; color: {MUTE};
  border-left: 4px solid {ACCENT}; padding-left: .55cm; white-space: nowrap; }}
.honesty .bd {{ color: {BUILT}; }} .honesty .sg {{ color: {SIGNAL}; }} .honesty .tg {{ color: {TARGET}; }}

/* cards */
.cards {{ display: flex; gap: 1.0cm; margin-top: .4cm; }}
.card {{ flex: 1; border: 1px solid {LINE}; border-top: 5px solid {ACCENT};
  border-radius: 8px; padding: .7cm .8cm; background: {PANEL}; }}
.card-tag {{ font-family: "Microsoft YaHei",sans-serif; font-size: 17px;
  color: {ACCENT_D}; font-weight: 700; letter-spacing: .5px; }}
.card-title {{ font-size: 30px; font-weight: 700; margin: .18cm 0 .3cm; }}
.card-body {{ font-size: 20px; line-height: 1.65; color: {INK}; }}

/* big bullet list */
.big-list {{ list-style: none; }}
.big-list li {{ position: relative; font-size: 22px; line-height: 1.55;
  padding: .3cm 0 .3cm 1.0cm; border-bottom: 1px solid {LINE}; }}
.big-list li:last-child {{ border-bottom: none; }}
.big-list li:before {{ content: ""; position: absolute; left: 0; top: .52cm;
  width: .42cm; height: .42cm; background: {ACCENT}; border-radius: 50% 50% 50% 0;
  transform: rotate(-45deg); }}

/* differentiation rows */
.rows {{ display: flex; flex-direction: column; gap: .5cm; }}
.row {{ border: 1px solid {LINE}; border-left: 5px solid {ACCENT}; border-radius: 6px;
  padding: .5cm .7cm; background: {PANEL}; }}
.row-h {{ font-size: 23px; font-weight: 700; margin-bottom: .12cm; }}
.row-b {{ font-size: 19px; line-height: 1.55; color: {INK}; }}

/* graded evidence */
.grade {{ margin-bottom: .4cm; }}
.grade-head {{ font-family: "Microsoft YaHei",sans-serif; font-size: 22px;
  font-weight: 700; padding: .12cm 0 .12cm; }}
.grade-head.bd {{ color: {BUILT}; }} .grade-head.sg {{ color: {SIGNAL}; }} .grade-head.tg {{ color: {TARGET}; }}
.grade ul {{ list-style: none; }}
.grade li {{ position: relative; font-size: 18px; line-height: 1.5;
  padding: .06cm 0 .06cm .55cm; }}
.grade li:before {{ content: ""; position: absolute; left: 0; top: .32cm;
  width: 6px; height: 6px; background: {MUTE}; border-radius: 50%; }}
.scope {{ display: block; font-size: 15px; color: {MUTE}; font-style: italic; }}

/* columns */
.cols {{ display: flex; gap: 1.2cm; }}
.col {{ flex: 1; }}
.col-h {{ font-family: "Microsoft YaHei",sans-serif; font-size: 19px; font-weight: 700;
  color: {ACCENT_D}; letter-spacing: .5px; margin-bottom: .3cm;
  border-bottom: 2px solid {ACCENT}; padding-bottom: .15cm; display: inline-block; }}
.ms {{ padding-left: .7cm; margin-top: .15cm; }}
.ms li {{ font-size: 21px; line-height: 1.55; margin-bottom: .3cm; }}

/* closing */
.assets {{ margin-top: .5cm; border: 1px solid {LINE}; border-radius: 8px;
  padding: .65cm .9cm; background: {PANEL}; }}
.assets-h {{ font-family: "Microsoft YaHei",sans-serif; font-size: 19px; font-weight: 700;
  color: {ACCENT_D}; margin-bottom: .3cm; }}
.assets-row {{ font-family: "Microsoft YaHei",sans-serif; font-size: 20px; line-height: 2.0; }}
.assets-row .k {{ display: inline-block; width: 5.6cm; color: {MUTE}; }}
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
