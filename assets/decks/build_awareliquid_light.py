# -*- coding: utf-8 -*-
"""Build the AwareLiquid "Investor Deck Light" as a styled HTML, ready for
Chrome headless --print-to-pdf.

Design: iOS / Apple "frosted glass" (glassmorphism) -- soft pastel gradient-mesh
backgrounds with translucent blurred white panels, rounded corners, light shadows.
Typography: Aeonik (display / Latin headings + wordmark; falls back to the
embedded Manrope, a free geometric grotesque, when Aeonik is not licensed on the
host) + Microsoft YaHei (微软雅黑) for Chinese body. Key terms bolded.

CONTENT is the honest, graded (Built / Early-signal / Target) 5-minute version,
with the ported narrative beats (memory crisis / KV-cache, microtubule
bio-inspiration) AND a new slide articulating the native capabilities a
Transformer lacks (continuous-time memory, spatial memory & reasoning,
imagination/rollout, goal self-modulation, predictive-coding self-correction,
continual learning) -- each honestly framed as architecture-native mechanism
(Built), with at-scale effectiveness graded per the evidence slide.
"""
import base64
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
FIG = ROOT / "assets/figures"
DECK = ROOT / "assets/decks"


def b64(path):
    return base64.b64encode(pathlib.Path(path).read_bytes()).decode()


LOGO = b64(FIG / "logo.png")
IMG_COST = b64(FIG / "cost_explosion.png")
IMG_MICRO = b64(FIG / "fig_microtubules.png")
IMG_ARCH = b64(FIG / "fig_architecture.png")
IMG_VIZ = b64(FIG / "fig_mtlnn_viz.png")
FONT_MANROPE = b64(DECK / "Manrope-variable.ttf")

# ---- iOS frosted-glass palette ----
INK = "#1d2440"         # deep navy ink (Apple-dark)
MUTE = "#5b6480"        # slate gray
ACCENT = "#0a84ff"      # iOS system blue
ACCENT2 = "#5e5ce6"     # iOS indigo
BUILT = "#11824a"       # green (Built)
SIGNAL = "#b5701a"      # amber (Early signal)
TARGET = "#7a3ff2"      # violet (Target)
GLASS = "rgba(255,255,255,.66)"
GLASS_SOFT = "rgba(255,255,255,.5)"
GLASS_BD = "rgba(255,255,255,.7)"

# display font stack (Latin): real Aeonik if licensed, else embedded Manrope
DISP = '"Aeonik","Manrope","Microsoft YaHei","Segoe UI",sans-serif'
HAN = '"Microsoft YaHei","Segoe UI",sans-serif'

SLIDES = []


def slide(html):
    SLIDES.append(html)


# 1. Title -----------------------------------------------------------------
slide(f"""
<section class="slide title">
  <div class="hero-glass">
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
      <b class="bd">Built</b> · <b class="sg">Early&nbsp;signal</b> · <b class="tg">Target</b>；无法通过技术尽调的数字，不写进本 deck。</div>
  </div>
</section>
""")

# 2. Memory crisis ---------------------------------------------------------
slide(f"""
<section class="slide">
  <div class="panel">
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
        <div class="fig-card"><img src="data:image/png;base64,{IMG_COST}" alt="cost"></div>
        <div class="fig-cap">概念示意：复杂度量级对比（asymptotic），非实测基准。</div>
      </div>
    </div>
  </div>
</section>
""")

# 3. Bio-inspiration -------------------------------------------------------
slide(f"""
<section class="slide">
  <div class="panel">
    <h2>生物学启示：从人脑到微管</h2>
    <div class="split">
      <div class="split-l">
        <ul class="big-list">
          <li><b>大脑不记录每个像素</b>　认知靠「工作记忆」滞存核心线索，靠「选择性遗忘」抛弃噪音。</li>
          <li><b>微管：神经计算的底层基座</b>　神经元内部的细胞骨架网络，以极高频率动态过滤、压缩信息。</li>
          <li><b>从生物到 LNN</b>　摒弃强制缓存历史，用类微管的<b>连续微分方程</b>，让信息随时间流动、留存与遗忘。</li>
        </ul>
      </div>
      <div class="split-r">
        <div class="fig-card"><img src="data:image/png;base64,{IMG_MICRO}" alt="microtubules"></div>
        <div class="fig-cap">受微管动态启发的「液态状态流」灵感链。</div>
      </div>
    </div>
  </div>
</section>
""")

# 4. Product matrix + market positioning ----------------------------------
slide(f"""
<section class="slide">
  <div class="panel">
    <h2>产品矩阵与市场定位</h2>
    <p class="lead">一条主线：<b>低成本 · 低幻觉 · 可私有化</b>。不在通用基准上正面拼参数，
    而是用架构差异，在两个具体市场把<b>「大材小用 + 数据出境 + 幻觉不可控」</b>的痛点接走。</p>
    <div class="cards">
      <div class="card">
        <div class="card-tag">产品线 O1 · 面向硬件</div>
        <div class="card-title">液态时序基座</div>
        <div class="card-body"><b>端侧 / 边缘的低成本、低功耗推理</b>。O(1) 工作记忆不随上下文爆显存，
        可在手机与边缘设备长驻。正面对标 edge 推理这一层——但<b>全栈开源可复现</b>。</div>
      </div>
      <div class="card">
        <div class="card-tag">产品线 M1 · 面向应用</div>
        <div class="card-title">自主认知智能体</div>
        <div class="card-body">在<b>高可靠 / 私有化 / 垂直窄域</b>场景，替代「凡事调用 DeepSeek / Gemini」的重方案：
        更便宜、数据不出境、自主闭环，架构内置误差自校正以<b>降低幻觉</b>。</div>
      </div>
    </div>
    <div class="posnote">定位说明：M1 <b>不</b>在通用 benchmark 上与前沿大模型正面对打；主打「在不需要前沿能力的场景里替换掉它们」。
    「低幻觉」为架构设计目标——误差自校正机制已落地（Built），规模化有效性见分级（Target）。</div>
  </div>
</section>
""")

# 5. NEW: capabilities Transformer lacks ----------------------------------
slide(f"""
<section class="slide">
  <div class="panel">
    <h2>Transformer 没有、我们<b>架构原生</b>具备的能力</h2>
    <p class="sublead">Transformer 是被动的「下一个词预测器」，下列能力靠提示词与外挂脚手架硬凑；
    在我们的架构里它们是<b>原生机制</b>——机制已落地，规模化有效性见「进展与证据」分级。</p>
    <div class="cap-grid">
      <div class="capx"><div class="cap-h">连续时间记忆 <span class="tag bd">机制</span></div>
        <div class="cap-b"><i>O(1)</i> 工作记忆衰减矩阵，而非 <i>O(N²)</i> KV 缓存——内存不随上下文爆炸。</div></div>
      <div class="capx"><div class="cap-h">空间记忆 & 空间推演 <span class="tag bd">机制</span><span class="tag tg">规模化待验</span></div>
        <div class="cap-b">把信息组织成可推演的空间结构，而非纯一维 token 序列。</div></div>
      <div class="capx"><div class="cap-h">想象推演（内部模拟）<span class="tag bd">机制</span><span class="tag tg">规模化待验</span></div>
        <div class="cap-b">行动前主动 rollout 未来、推演后果，而非被动接龙下一个 token。</div></div>
      <div class="capx"><div class="cap-h">目标自调制 <span class="tag bd">机制</span></div>
        <div class="cap-b">自主设定目标并据此调制行为，而非纯指令驱动的被动应答。</div></div>
      <div class="capx"><div class="cap-h">误差自校正 <span class="tag bd">机制</span></div>
        <div class="cap-b">层级预测编码自带误差回传与自校正，原生对抗漂移。</div></div>
      <div class="capx"><div class="cap-h">持续学习 / 抗遗忘 <span class="tag sg">早期信号</span></div>
        <div class="cap-b">无需重训持续吸收新数据；小规模探针上遗忘率低于基线（带误差棒）。</div></div>
    </div>
  </div>
</section>
""")

# 6. Differentiation -------------------------------------------------------
slide(f"""
<section class="slide">
  <div class="panel">
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
        <div class="fig-card"><img src="data:image/png;base64,{IMG_ARCH}" alt="architecture"></div>
        <div class="fig-cap">MT-LNN 架构示意。</div>
      </div>
    </div>
  </div>
</section>
""")

# 7. NEW: architecture visualization --------------------------------------
slide(f"""
<section class="slide">
  <div class="panel">
    <h2>架构可视化：把 Transformer 的 FFN 换成<b>类脑核心</b></h2>
    <div class="split viz-split">
      <div class="split-l">
        <div class="pipe">
          <div class="pstep"><span class="pn">1</span><b>Tokens → Embedding</b>　与标准 GPT 一致的输入/位置嵌入。</div>
          <div class="pstep"><span class="pn">2</span><b>Causal Self-Attention</b>　保留注意力做 token 间混合。</div>
          <div class="pstep hot"><span class="pn">3</span><b>MT-DL（13 Protofilaments × 5 液态时间尺度）</b>
            用类微管动态层替换 FFN——连续时间、多时间尺度的状态流，是架构核心创新。</div>
          <div class="pstep hot"><span class="pn">4</span><b>GWTB 全局工作空间瓶颈</b>
            832 → 104 → 832 编解码，核心以 <i>O(1)</i> 更新维持工作记忆，而非 <i>O(N²)</i> KV 缓存。</div>
          <div class="pstep"><span class="pn">5</span><b>Output Projection → Softmax</b>　标准输出头，整链可与 GPT 对照。</div>
        </div>
        <div class="vnote">右图为仓库内 <b>llm-viz</b> 工具直接渲染的真实结构视图（非概念示意）；
        13 通道与 5 时间尺度对应代码中的 <i>n_heads=13</i> 与液态时间常数配置。</div>
      </div>
      <div class="split-r">
        <div class="fig-card dark"><img src="data:image/png;base64,{IMG_VIZ}" alt="mtlnn 3d"></div>
        <div class="fig-cap">MT-LNN 3D 结构：Self-Attention + MT-DL(13×5) ×N → GWTB(O(1))。</div>
      </div>
    </div>
  </div>
</section>
""")

# 8. NEW: Liquid AI competitor comparison ---------------------------------
slide(f"""
<section class="slide">
  <div class="panel">
    <h2>竞品对照：Liquid AI vs <b>AwareLiquid</b></h2>
    <p class="sublead">同样押注「Transformer 之外的连续时间路线」，但<b>架构轴心与目标市场不同</b>。
    Liquid AI 更成熟、资金更充足；我们押的是另一条更偏<b>类脑结构 + 安全可审计 + 全栈开源</b>的差异化轴。</p>
    <div class="cmp">
      <div class="cmp-row cmp-head">
        <div class="cmp-k">维度</div>
        <div class="cmp-a">Liquid AI</div>
        <div class="cmp-b">AwareLiquid（我们）</div>
      </div>
      <div class="cmp-row">
        <div class="cmp-k">阶段 / 资金</div>
        <div class="cmp-a">~$2.35B 估值，$250M A 轮（AMD 领投，2024）</div>
        <div class="cmp-b">早期；125M 基座跑通，全栈开源可复现</div>
      </div>
      <div class="cmp-row">
        <div class="cmp-k">核心架构</div>
        <div class="cmp-a">LFM2：卷积 + GQA 混合；LIV 算子 / STAR 架构搜索；从 7B 蒸馏</div>
        <div class="cmp-b">13 平行通道（微管启发）× 5 液态时间尺度 + GWTB 瓶颈</div>
      </div>
      <div class="cmp-row">
        <div class="cmp-k">记忆机制</div>
        <div class="cmp-a">高吞吐近似；面向通用 edge 生成</div>
        <div class="cmp-b"><i>O(1)</i> 工作记忆 + 层级预测编码误差自校正</div>
      </div>
      <div class="cmp-row">
        <div class="cmp-k">安全 / 可靠</div>
        <div class="cmp-a">通用基座，安全交由上层</div>
        <div class="cmp-b">架构内置 failsafe 优雅降级 + 麻醉态可验证探针</div>
      </div>
      <div class="cmp-row">
        <div class="cmp-k">目标市场</div>
        <div class="cmp-a">通用端侧基座（350M–1.2B）</div>
        <div class="cmp-b">垂直窄域 / 工控 / 具身 / 高合规私有化</div>
      </div>
      <div class="cmp-row">
        <div class="cmp-k">开放程度</div>
        <div class="cmp-a">权重开放，核心研究框架闭源</div>
        <div class="cmp-b">代码 + 模型 + 1083 项测试全栈开源</div>
      </div>
    </div>
    <div class="posnote">诚实说明：Liquid AI 在规模、工程成熟度与商业化上<b>远领先于我们</b>。
    我们不主张性能反超，而是押注一条不同的架构轴——<b>类脑可解释结构 + 原生安全兜底 + 全栈可独立验证</b>，
    在通用基座覆盖不到的高可靠垂直场景里建立壁垒。</div>
  </div>
</section>
""")

# 9. Evidence, graded ------------------------------------------------------
slide(f"""
<section class="slide">
  <div class="panel">
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
        <li>跨域灾难性遗忘对照；端侧路径用蒸馏让小模型在<b>特定窄任务</b>对标更大模型。</li>
      </ul>
    </div>
  </div>
</section>
""")

# 8. Go-to-market ----------------------------------------------------------
slide(f"""
<section class="slide">
  <div class="panel">
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
  </div>
</section>
""")

# 9. Investment logic ------------------------------------------------------
slide(f"""
<section class="slide">
  <div class="panel closing-panel">
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
  </div>
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
@font-face {{ font-family: "Manrope"; font-weight: 200 800; font-style: normal;
  src: url(data:font/ttf;base64,{FONT_MANROPE}) format("truetype"); }}
@page {{ size: 33.867cm 19.05cm; margin: 0; }}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body {{ color: {INK};
  font-family: {HAN}; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}

.slide {{
  position: relative; width: 33.867cm; height: 19.05cm;
  padding: 1.4cm 1.6cm 1.5cm 1.6cm; overflow: hidden;
  page-break-after: always; display: flex; flex-direction: column;
  background:
    radial-gradient(42% 55% at 14% 16%, rgba(122,168,255,.55), transparent 62%),
    radial-gradient(46% 58% at 88% 12%, rgba(199,155,255,.50), transparent 62%),
    radial-gradient(50% 62% at 78% 92%, rgba(127,240,208,.45), transparent 62%),
    radial-gradient(44% 56% at 8% 90%, rgba(255,182,193,.38), transparent 62%),
    linear-gradient(135deg, #eef1f7 0%, #e9edf5 100%);
}}
.slide:last-child {{ page-break-after: auto; }}

/* frosted-glass panel */
.panel {{
  position: relative; z-index: 1; flex: 1; display: flex; flex-direction: column;
  border-radius: 26px; padding: 1.0cm 1.3cm 1.1cm;
  background: {GLASS}; backdrop-filter: blur(22px) saturate(140%);
  -webkit-backdrop-filter: blur(22px) saturate(140%);
  border: 1px solid {GLASS_BD};
  box-shadow: 0 22px 60px rgba(31,45,90,.16), inset 0 1px 0 rgba(255,255,255,.85);
}}

h2 {{ font-family: {DISP}; font-weight: 700; font-size: 38px; color: {INK};
  letter-spacing: .2px; margin-bottom: .55cm; padding-bottom: .3cm; position: relative; }}
h2 b {{ color: {ACCENT}; font-weight: 800; }}
h2:after {{ content: ""; position: absolute; left: 0; bottom: 0; width: 2.8cm; height: 4px;
  background: linear-gradient(90deg, {ACCENT}, {ACCENT2}); border-radius: 3px; }}
.h2-note {{ font-size: 20px; font-weight: 400; color: {MUTE}; }}

.lead {{ font-size: 25px; line-height: 1.75; color: {INK}; margin-bottom: .5cm; }}
.lead.big {{ font-size: 27px; line-height: 1.85; }}
.sublead {{ font-size: 20px; line-height: 1.65; color: {MUTE}; margin-bottom: .5cm; }}
.lead b, .row-b b, li b, .card-body b, .callout b, .cap-b b, .sublead b, .sub b {{ color: {ACCENT}; font-weight: 700; }}

/* footline + logo (over gradient, outside panel) */
.footline {{ position: absolute; left: 0; right: 0; bottom: .5cm; z-index: 2;
  display: flex; font-family: {DISP}; font-size: 12px; color: {MUTE}; }}
.fl {{ flex: 1; }}
.fl-l {{ padding-left: 1.7cm; letter-spacing: 1px; }}
.fl-c {{ text-align: center; }}
.fl-r {{ text-align: right; padding-right: 1.7cm; letter-spacing: .5px; }}
.logo {{ position: absolute; top: .5cm; right: .55cm; height: 1.0cm; opacity: .9; z-index: 3; }}

/* split layout (text + figure) */
.split {{ display: flex; gap: 1.0cm; flex: 1; align-items: center; }}
.split-l {{ flex: 1.15; }}
.split-r {{ flex: .95; text-align: center; }}
.fig-card {{ background: rgba(255,255,255,.8); border: 1px solid {GLASS_BD};
  border-radius: 16px; padding: .35cm; box-shadow: 0 10px 28px rgba(31,45,90,.12); }}
.fig-card img {{ max-width: 100%; max-height: 10.6cm; display: block; margin: 0 auto; border-radius: 8px; }}
.fig-cap {{ font-size: 15px; color: {MUTE}; font-style: italic; margin-top: .25cm; }}
.callout {{ margin-top: .45cm; font-size: 20px; line-height: 1.6; color: {INK};
  background: rgba(10,132,255,.10); border-left: 4px solid {ACCENT};
  border-radius: 0 12px 12px 0; padding: .4cm .6cm; }}

/* title hero */
.title {{ align-items: stretch; }}
.hero-glass {{ position: relative; z-index: 1; flex: 1; display: flex; flex-direction: column;
  justify-content: center; border-radius: 30px; padding: 1.6cm 2.0cm;
  background: rgba(255,255,255,.55); backdrop-filter: blur(26px) saturate(150%);
  -webkit-backdrop-filter: blur(26px) saturate(150%);
  border: 1px solid {GLASS_BD};
  box-shadow: 0 26px 70px rgba(31,45,90,.18), inset 0 1px 0 rgba(255,255,255,.9); }}
.brand-mark {{ font-family: {DISP}; font-weight: 600; font-size: 28px; letter-spacing: 5px;
  color: {ACCENT}; margin-bottom: .35cm; }}
.title h1 {{ font-family: {DISP}; font-size: 92px; font-weight: 800; letter-spacing: -.5px;
  line-height: 1.0;
  background: linear-gradient(120deg, {INK} 0%, {ACCENT2} 60%, {ACCENT} 100%);
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }}
.title .sub {{ font-size: 30px; line-height: 1.5; margin-top: .4cm; color: {INK}; }}
.title .sub2 {{ font-size: 21px; color: {MUTE}; margin-top: .12cm; }}
.title-meta {{ margin-top: .7cm; font-size: 19px; line-height: 1.85; font-family: {DISP}; color: {INK}; }}
.title-meta .k {{ display: inline-block; width: 4.0cm; color: {ACCENT}; font-weight: 700; }}
.honesty {{ margin-top: .7cm; font-size: 16px; line-height: 1.55; color: {MUTE};
  border-left: 4px solid {ACCENT}; padding-left: .5cm; white-space: nowrap; }}
.honesty .bd {{ color: {BUILT}; }} .honesty .sg {{ color: {SIGNAL}; }} .honesty .tg {{ color: {TARGET}; }}

/* product cards */
.cards {{ display: flex; gap: 1.0cm; margin-top: .35cm; }}
.card {{ flex: 1; border-radius: 18px; padding: .75cm .85cm;
  background: rgba(255,255,255,.6); border: 1px solid {GLASS_BD};
  border-top: 4px solid {ACCENT}; box-shadow: 0 12px 30px rgba(31,45,90,.10); }}
.card-tag {{ font-family: {DISP}; font-size: 17px; color: {ACCENT}; font-weight: 700; letter-spacing: .5px; }}
.card-title {{ font-size: 30px; font-weight: 700; margin: .18cm 0 .3cm; }}
.card-body {{ font-size: 19px; line-height: 1.62; color: {INK}; }}
.posnote {{ margin-top: .55cm; font-size: 16px; line-height: 1.55; color: {MUTE};
  background: rgba(122,63,242,.08); border-left: 4px solid {TARGET};
  border-radius: 0 12px 12px 0; padding: .4cm .6cm; }}
.posnote b {{ color: {INK}; }}

/* capabilities grid */
.cap-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: .55cm; flex: 1; align-content: center; }}
.capx {{ border-radius: 16px; padding: .55cm .65cm; background: rgba(255,255,255,.6);
  border: 1px solid {GLASS_BD}; box-shadow: 0 10px 26px rgba(31,45,90,.09); }}
.cap-h {{ font-family: {DISP}; font-size: 20px; font-weight: 700; color: {INK}; margin-bottom: .2cm; line-height: 1.3; }}
.cap-b {{ font-size: 16.5px; line-height: 1.55; color: {INK}; }}
.tag {{ display: inline-block; font-family: {DISP}; font-size: 12px; font-weight: 700;
  padding: 1px 8px; border-radius: 20px; margin-left: 6px; vertical-align: middle; }}
.tag.bd {{ color: {BUILT}; background: rgba(17,130,74,.12); }}
.tag.sg {{ color: {SIGNAL}; background: rgba(181,112,26,.14); }}
.tag.tg {{ color: {TARGET}; background: rgba(122,63,242,.12); }}

/* big bullet list */
.big-list {{ list-style: none; }}
.big-list li {{ position: relative; font-size: 22px; line-height: 1.55;
  padding: .28cm 0 .28cm 1.0cm; border-bottom: 1px solid rgba(31,45,90,.10); }}
.big-list li:last-child {{ border-bottom: none; }}
.big-list li:before {{ content: ""; position: absolute; left: 0; top: .5cm;
  width: .4cm; height: .4cm; background: linear-gradient(135deg, {ACCENT}, {ACCENT2});
  border-radius: 50% 50% 50% 0; transform: rotate(-45deg); }}

/* differentiation rows */
.rows {{ display: flex; flex-direction: column; gap: .45cm; }}
.row {{ border-radius: 14px; padding: .5cm .7cm; background: rgba(255,255,255,.6);
  border: 1px solid {GLASS_BD}; border-left: 4px solid {ACCENT}; box-shadow: 0 8px 22px rgba(31,45,90,.08); }}
.row-h {{ font-size: 22px; font-weight: 700; margin-bottom: .1cm; }}
.row-b {{ font-size: 19px; line-height: 1.55; color: {INK}; }}

/* graded evidence */
.grade {{ margin-bottom: .35cm; }}
.grade-head {{ font-family: {DISP}; font-size: 22px; font-weight: 700; padding: .1cm 0; }}
.grade-head.bd {{ color: {BUILT}; }} .grade-head.sg {{ color: {SIGNAL}; }} .grade-head.tg {{ color: {TARGET}; }}
.grade ul {{ list-style: none; }}
.grade li {{ position: relative; font-size: 18px; line-height: 1.5; padding: .05cm 0 .05cm .55cm; }}
.grade li:before {{ content: ""; position: absolute; left: 0; top: .32cm; width: 6px; height: 6px;
  background: {MUTE}; border-radius: 50%; }}
.scope {{ display: block; font-size: 15px; color: {MUTE}; font-style: italic; }}

/* columns */
.cols {{ display: flex; gap: 1.2cm; }}
.col {{ flex: 1; }}
.col-h {{ font-family: {DISP}; font-size: 19px; font-weight: 700; color: {ACCENT};
  letter-spacing: .5px; margin-bottom: .3cm; padding-bottom: .12cm;
  border-bottom: 2px solid rgba(10,132,255,.4); display: inline-block; }}
.ms {{ padding-left: .7cm; margin-top: .15cm; }}
.ms li {{ font-size: 21px; line-height: 1.55; margin-bottom: .3cm; }}

/* closing */
.closing-panel {{ justify-content: center; }}
.assets {{ margin-top: .5cm; border-radius: 16px; padding: .65cm .9cm;
  background: rgba(255,255,255,.6); border: 1px solid {GLASS_BD}; box-shadow: 0 10px 26px rgba(31,45,90,.09); }}
.assets-h {{ font-family: {DISP}; font-size: 19px; font-weight: 700; color: {ACCENT}; margin-bottom: .3cm; }}
.assets-row {{ font-family: {DISP}; font-size: 20px; line-height: 1.95; }}
.assets-row .k {{ display: inline-block; width: 5.6cm; color: {MUTE}; }}

/* architecture viz slide */
.viz-split {{ align-items: stretch; }}
.fig-card.dark {{ background: #1a1a2e; border-color: rgba(120,140,200,.35); }}
.fig-card.dark img {{ max-height: 11.4cm; border-radius: 10px; }}
.pipe {{ display: flex; flex-direction: column; gap: .3cm; }}
.pstep {{ position: relative; font-size: 18px; line-height: 1.5; color: {INK};
  padding: .32cm .55cm .32cm 1.15cm; border-radius: 12px; background: rgba(255,255,255,.55);
  border: 1px solid {GLASS_BD}; box-shadow: 0 6px 18px rgba(31,45,90,.07); }}
.pstep.hot {{ background: rgba(94,92,230,.10); border-left: 4px solid {ACCENT2}; }}
.pstep b {{ color: {INK}; }}
.pstep .pn {{ position: absolute; left: .42cm; top: .32cm; width: .56cm; height: .56cm; line-height: .56cm;
  text-align: center; font-family: {DISP}; font-size: 13px; font-weight: 800; color: #fff;
  background: linear-gradient(135deg, {ACCENT}, {ACCENT2}); border-radius: 50%; }}
.vnote {{ margin-top: .4cm; font-size: 15.5px; line-height: 1.55; color: {MUTE};
  border-left: 4px solid {ACCENT}; padding-left: .5cm; }}
.vnote b {{ color: {INK}; }}

/* competitor comparison table */
.cmp {{ display: flex; flex-direction: column; gap: 4px; margin-top: .2cm; }}
.cmp-row {{ display: flex; gap: 6px; }}
.cmp-row > div {{ border-radius: 10px; padding: .28cm .5cm; font-size: 16.5px; line-height: 1.42; }}
.cmp-k {{ flex: .8; font-family: {DISP}; font-weight: 700; color: {INK};
  background: rgba(255,255,255,.45); border: 1px solid {GLASS_BD}; display: flex; align-items: center; }}
.cmp-a {{ flex: 1.35; color: {INK}; background: rgba(120,130,160,.10); border: 1px solid rgba(120,130,160,.2); }}
.cmp-b {{ flex: 1.35; color: {INK}; background: rgba(10,132,255,.10); border: 1px solid rgba(10,132,255,.22); }}
.cmp-b i, .cmp-a i {{ font-style: normal; font-weight: 700; }}
.cmp-head > div {{ font-family: {DISP}; font-weight: 800; font-size: 18px; }}
.cmp-head .cmp-a {{ color: {MUTE}; }}
.cmp-head .cmp-b {{ color: {ACCENT}; }}
"""

HTML = f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>AwareLiquid — Investor Deck Light</title>
<style>{CSS}</style></head>
<body>
{''.join(footed)}
</body></html>"""

out = DECK / "AwareLiquid_Investor_Deck_Light.html"
out.write_text(HTML, encoding="utf-8")
print("wrote", out, len(HTML), "bytes,", N, "slides")
