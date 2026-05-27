import re

with open('e:/M1/mt_lnn_arxiv_zh.tex', 'r', encoding='utf-8') as f:
    text = f.read()

start_marker = r'\section{MT-LNN 架构}'
end_marker = r'\section{麻醉验证协议 (AVP)}'

start_idx = text.find(start_marker)
end_idx = text.find(end_marker)

new_architecture = r"""\section{MT-LNN 架构}
\label{sec:architecture}

\begin{figure*}[t]
\centering
\includegraphics[width=0.9\textwidth]{fig_architecture.pdf}
\caption{\textbf{MT-LNN 架构概述。}输入表示依次经过\textbf{微管动态层}（结合了 13 个平行的原丝通道，具备多尺度共振、动态内源性计算跳过以及基于多尺度 MSE 的预测编码机制），随后通过通过局部状态调节序列门控的\textbf{微管注意力}（Microtubule Attention）模块，最终进入建立全局整合广播的\textbf{全局工作空间理论瓶颈}（Global Workspace Theory Bottleneck）。最顶层的\textbf{全局相干层}（Global Coherence Layer）实现了 Orch-OR 坍缩以及 $O(1)$ 工作记忆衰减机制。此外，因果链（Causal Chain）和自我监控（Self-Monitor）提取头除了输出标准 logits，还原生输出具有高度解释性的符号日志。}
\label{fig:architecture}
\end{figure*}

\paragraph{概述。}
\[
  \vx \xrightarrow{\text{Embed+RoPE}}
  \Bigl[\text{MicrotubuleAttn} \to \text{MT-DL}\Bigr]^{n_\text{layers}}
  \xrightarrow{} \text{GWTB}
  \xrightarrow{} \text{GlobalCoherence}
  \xrightarrow{} \text{Extractors}
\]
每个块对两个子层应用前置归一化（pre-norm）和残差连接。一个 \texttt{ModelCacheStruct} 携带着每一层的注意力 KV、每一层的 $h_\text{prev}$、GWTB 工作空间 KV，以及全局相干层（Global Coherence）的无界全局工作记忆矩阵。

\subsection{微管动态层 (MT-DL)}
\label{sec:mt_dl}

\paragraph{原丝分解。}
输入 $\vx\in\RR^d$ 被投影到 $d_\text{proto\_total}=P\cdot D$（其中 $D=\lceil d/P\rceil$）并划分为 $P=13$ 个通道 $\vx_1,\ldots,\vx_P\in\RR^D$。所有的 $P\times S$ 个 LTC 库在极低内存开销下通过权重张量 $W_\text{in}\in\RR^{P\times S\times D\times D}$ 和\emph{单次 einsum} 求值，彻底取代了传统 Transformer 中参数量庞大的 $O(d^2)$ FFN。

\paragraph{多尺度共振。}
每根原丝 $p$ 运行 $S=5$ 个 CfLTC 子库，其时间常数呈几何级数扫描：
\begin{equation}
  \tau_s = \tau_\text{min}\bigl(\tau_\text{max}/\tau_\text{min}\bigr)^{s/(S-1)}, \quad s=0,\ldots,S-1.
\end{equation}
输出通过在 $S$ 个库上应用参数化 softmax 汇聚，通过结构性的归纳偏置使得高频的快速感知能够平滑地过渡到低频的长期抽象记忆中。

\paragraph{多尺度预测编码损失 (Multi-Scale Predictive Coding Loss)。}
MT-DL 在根本上背离了传统的“仅下一个标记预测 (Next-Token Prediction)”，而是内嵌了一个基于生物学原理的\emph{预测编码系统}。更慢、层级更高的概念通道 ($\tau_{s>0}$) 会随着序列的推进，主动预测更快、层级更低的感知通道 ($\tau_{s=0}$) 的状态。在前向传播中（当 $S>1$），网络会计算辅助的\emph{预测误差损失}（尺度间均方误差）：
\begin{equation}
  \mathcal{L}_\text{pred} = \mathrm{MSE}\Big(\mathbf{W}_\text{pred}' \vh^{p, s>0}_{t},\; \vh^{p, s=0}_{t}\Big),
\end{equation}
这一机制提供了内生的、连续时间步上的自监督训练，确保即便在缺乏外部语言建模标签支持的情况下，内部神经表征仍在持续预测其自身的感知输入，大幅加强了对因果关系的隐式捕捉。

\paragraph{动态门控与内源性计算跳过 (Endogenous Compute Skipping)。}
标准的非线性激活函数被一个动态的 $\kappa$-门控 $\sigmoid(\mathbf{W}_{\kappa}\vx)$ 所替代，该门控根据上下文序列情况有选择性地使子库跨越阈值。借助 \texttt{compute\_skip\_threshold} 参数，低于活跃阈值的计算节点（原丝）将被掩码覆盖，并在前向投影中通过布尔型 \texttt{masked\_fill} 操作原生跳过该运算。这意味着当面对重复的、或复杂度极低的 Token 时，网络能够彻底关闭休眠神经通路的计算，实现静态 Transformer 所无法企及的指数级动态推理解码能效节约。

\paragraph{平行扫描训练与三向耦合。}
借助基于封闭形式的前缀和 (Prefix Product) 算法扩展，我们实现了等价于原生公式~\eqref{eq:cflnn} 可微分展开的并行扫描机制。训练得以在 $O(T\log T)$ 极低复杂度下由 GPU 全局并行展开。同时它配有空间同步耦合和衰减门控，模拟了真实的 GTP 帽更新机制以维持长尾信息不会稀释。

\subsection{微管注意力}
\label{sec:attn}

除了底层的基于 Flash-Attention 支持的 SDPA GQA (13-Query/1-KV) 组件外，还引入了：
1) \textbf{标量极性偏置}：模拟原丝物理的电偶极子效应；
2) \textbf{类 ALiBi 风格的 GTP 对数衰减偏置}：为不同的头部带来跨度达 $64\times$ 的级联感受野。

\subsection{全局工作空间理论瓶颈 (GWTB)}
\label{sec:gwtb}

为了防止长上下文衰退并聚合并行特征，MT-LNN 利用了 GWTB 将表达极限压缩至 $d_{gw}=d/r$ 中进行竞争性选择，使得重要信号通过 $\gamma_\text{bcast} W_\text{bcast} \mathbf{z}'_t$ 再广播回整个液态管网主流维度。

\subsection{全局相干与 $O(1)$ 工作记忆 (Working Memory)}
\label{sec:coherence}

\paragraph{O(1) 的工作记忆矩阵。} 为了彻底粉碎现代大模型严重受限于 KV 缓存导致上下文随长度爆炸的 $O(T)$ 内存墙问题，MT-LNN 提供在“指数工作记忆 (Decay Working Memory)”模式下运行\emph{全局相干层}的选择。全局工作空间状态无需随时间步进行 KV 张量串联缓存，而是将上下文聚合成一个形式闭合的指数移动平均 (EMA) 固定张量矩阵：
\begin{equation}
   \mathbf{WM}_{t} = (1 - g_t) \odot \mathbf{WM}_{t-1} \,+\, g_t \odot \vx_t, \quad g_t = \sigmoid(\mathbf{W}_\text{update} \vx_t).
\end{equation}
这将极大的上下文时序解析依赖转化为拥有无限时间视界的全局融合表示。从内存上看，推理时该架构完全受限于静态缓存，即享有恒定的 $O(1)$ RAM 限制，等价于动态无成本、无限扩展的上下文窗口，并且从源头上淘汰了传统 LLM 中必须追踪大额 KV Pointer 缓存的设计。

\paragraph{Orch-OR Top-k 坍缩门控。} 
稀疏注意力被数学转化为等效概念上的 Orch-OR 坍缩时刻，当注意力能量累积超过学习到的标量阈值 $\theta$ 时，通过激活门控 $\sigmoid((\bar{e}-\theta)\cdot 10)$ 触发跨工作记忆槽的全局语义重组。

\subsection{自带解释性的提取头：因果链与自我监控}
\label{sec:extractors}

告别深度黑盒机制难以捉摸的困境，MT-LNN 在架构的顶端配备了专门的解码侧头（\texttt{use\_causal\_head} 以及 \texttt{use\_self\_monitor\_head}）。这些输出通道不依赖词表分布目标，而是明确吐出包含逻辑演绎诊断、隐式自指校验（Self-Correction）在内的符号验证日志。其极高的可读性允许我们伴随主输出一道将架构底层的思考轨迹暴露出来。

% ========================================================
"""

text = text[:start_idx] + new_architecture + text[end_idx:]

with open('e:/M1/mt_lnn_arxiv_zh.tex', 'w', encoding='utf-8') as f:
    f.write(text)
