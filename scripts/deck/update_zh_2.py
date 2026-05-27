import re

with open('e:/M1/mt_lnn_arxiv_zh.tex', 'r', encoding='utf-8') as f:
    content = f.read()

new_abstract = r"""% ---- Abstract ----
\begin{abstract}
我们引入了 \textbf{MT-LNN} (\textbf{M}icro\textbf{t}ubule-Inspired \textbf{L}iquid \textbf{N}eural \textbf{N}etwork, 受微管启发的液态神经网络)，该架构彻底打破了传统的“下一个词预测 (Next-Token Prediction)”范式，交付了一个基于生物启发的\emph{预测编码系统 (Predictive Coding System)}。融合了神经元微管 (MTs)、闭式液态时间常数网络 (CfLTC) 以及意识的全局工作区理论 (GWT)，MT-LNN 将标准 Transformer 中的前馈网络 (FFN) 替换为\emph{微管动态层} (MT-DL) —— 13 个平行的 CfLTC 通道，具有多尺度共振与动态 $\kappa$ 门控实现的内源性计算跳过。MT-LNN 的核心是一个统一的\emph{多尺度预测编码损失}，较慢的 $\tau$ 抽象通道通过尺度内 MSE 自我监督较快的感知通道。为打破 $O(T)$ 的上下文内存墙，该架构通过指数衰减状态引入了 \emph{$O(1)$ 工作记忆缓存} 机制。此外，状态表示可以通过内置的\emph{因果链 (Causal Chain)} 和\emph{自我监控 (Self-Monitor)} 提取头映射为具有强解释性的符号日志。经由我们首创的\emph{麻醉验证协议} (AVP) 测试，MT-LNN 在 125M 参数级别相比同等 Transformer 困惑度降低了 $\approx$14.7\%，信息整合度 ($\phihat$) 提高了 $2.2\times$ 倍，并且凭借跳过门控与内存界限在推理芯片上实现了指数级的成本耗能节约。

\medskip
\noindent\textbf{关键词:} 预测编码, 液态神经网络, 微管, $O(1)$ 内存, 计算跳过, 全局工作区理论.
\end{abstract}"""

new_contrib = r"""\medskip\noindent\textbf{主要贡献。} MT-LNN 做出了如下核心贡献：
\begin{enumerate}[label=\textbf{C\arabic*},leftmargin=*]
  \item \textbf{微管动态层 (MT-DL).} 13 根原纤维并行的 CfLTC 设计，具备 $P\!\times\!S$ 多尺度共振库。采用了动态门控实现\textbf{内源性计算跳过 (Endogenous Compute Skipping)}，极大削减了休眠通道的计算开销，并集成了一个\textbf{多尺度预测编码损失}，使得较慢的抽象通道能够通过尺度间均方误差 (MSE) 自我监督较快的感知通道。
  \item \textbf{$O(1)$ 的工作记忆矩阵.} 整体相干状态通过配置的衰减率进行指数移动平均滚动，彻底消除了传统 Transformer Key-Value 缓存的 $O(T)$ 内存墙开销。
  \item \textbf{透明的因果与监控提取.} 取代了黑盒的单一输出层，将提取头映射到明确的因果链日志和自我监控逻辑中，极大增强了模型行为的解释性。
  \item \textbf{微管注意力.} 基于 SDPA 的 GQA（13Q/1KV），附带标量极性偏置以及具有 ALiBi 风格的 per-head GTP 级联衰减。
  \item \textbf{GWTB + 整体相干层.} 容量受限的全局工作区，与附带稀疏门控坍缩机制的相干层协同工作。
  \item \textbf{麻醉验证协议 (AVP).} 首个通过计算模拟临床麻醉效果的协议，其引入了基于 kNN 的 $\phihat$ 近似计算。
  \item \textbf{工程级实现.} Flash-Attention, 双状态持久化缓存, 谱整合提取指标 ($\phig$), 以及用于 $O(\log T)$ 时间复杂度内的并行扫描。
\end{enumerate}"""

content = re.sub(r'% ---- Abstract ----.*?\\end\{abstract\}', new_abstract, content, flags=re.DOTALL)
content = re.sub(r'\\medskip\\noindent\\textbf\{主要贡献。\} MT-LNN 做出了如下贡献：.*?\\end\{enumerate\}', new_contrib, content, flags=re.DOTALL)

with open('e:/M1/mt_lnn_arxiv_zh.tex', 'w', encoding='utf-8') as f:
    f.write(content)
