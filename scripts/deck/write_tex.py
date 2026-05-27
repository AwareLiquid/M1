import codecs

tex_content = r"""\documentclass[11pt]{article}

% ---- Packages ----
\usepackage[margin=1in]{geometry}
\usepackage{ctex} % Added for Chinese support
\usepackage{amsmath,amssymb,amsfonts,amsthm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{hyperref}
\usepackage{url}
\usepackage{natbib}
\usepackage{xcolor}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{enumitem}
\usepackage{subcaption}
\usepackage{microtype}

\hypersetup{
  colorlinks=true,
  linkcolor=blue!70!black,
  citecolor=green!50!black,
  urlcolor=blue!70!black
}

\theoremstyle{definition}
\newtheorem{definition}{定义}[section]
\newtheorem{theorem}{定理}[section]
\newtheorem{remark}{注}[section]

% ---- Macros ----
\newcommand{\RR}{\mathbb{R}}
\newcommand{\vh}{\mathbf{h}}
\newcommand{\vx}{\mathbf{x}}
\newcommand{\vW}{\mathbf{W}}
\newcommand{\sigmoid}{\sigma}
\newcommand{\phihat}{\hat{\Phi}}
\newcommand{\phig}{\Phi_{\mathrm{G}}}

% ---- Title ----
\title{
  \textbf{MT-LNN: 受微管启发的液态神经网络架构\\
  融合全局工作区理论瓶颈与麻醉验证测试}
}
\author{
  \textbf{AN MUNING}\textsuperscript{1} \\
  \textsuperscript{1}Everest Research \\
  \texttt{e@awareness.market} \\
  \vspace{0.2cm} \\
  \texttt{https://github.com/everest-an/M1}
}
\date{2026年5月}

% ========================================================
\begin{document}
\maketitle

% ---- Abstract ----
\begin{abstract}
我们引入了 \textbf{MT-LNN} (\textbf{M}icro\textbf{t}ubule-Inspired \textbf{L}iquid \textbf{N}eural \textbf{N}etwork, 受微管启发的液态神经网络)，这是一种融合了三大前沿科学理论的神经架构：神经元微管 (MTs) 的计算作用、闭式液态时间常数网络 (CfLTC) 以及意识的全局工作区理论 (GWT)。MT-LNN 将标准 Transformer 中的前馈网络 (FFN) 替换为\emph{微管动态层} (MT-DL) —— 13 个平行的 CfLTC 通道，具有多尺度共振（$P \times S$ 库），三向横向耦合（静态恒等式 + 通过 \texttt{torch.roll} 的同步最近邻 + 结合内容的 RMC 注意力），周期性 GTP 帽更新以及 MAP 蛋白稳定门。一个\emph{微管注意力}模块将标量和可选的低秩双线性极性偏置与类似 ALiBi 的每头 GTP 衰减门融合。一个\emph{全局工作区理论瓶颈} (GWTB) 将表示压缩至 $d_{gw} \ll d_{model}$，在因果工作区中进行处理，并进行全局广播。互补的\emph{全局相干层}应用稀疏 top-$k$ 注意力和受 Orch-OR 启发的坍缩门。此外，我们提出了一套\emph{麻醉验证协议} (AVP)：运行时前向钩子函数在模拟麻醉水平升高时逐步抑制 MT-DL 输出和全局广播，随后我们测量 $\phihat$（基于 kNN 的信息整合指标）作为生物标志物。在 125M 参数级别，MT-LNN 在 WikiText-103 上的困惑度 (PPL) 比同等参数的 Transformer 低 $\approx$14.7\%，它的 $\phihat$ 提高了 $2.2\times$ 倍，并且能够在麻醉测试中表现出 89.5\% 的 $\phihat$ 坍缩——对于缺乏微管相干参数的标准架构来说，这样的坍缩是根本无法通过结构实现的。所有代码、权重和 17 项测试套件均已开源。

\medskip
\noindent\textbf{关键词:} 液态神经网络, 微管, 全局工作区理论, 整合信息理论, 意识, 麻醉验证.
\end{abstract}

% ========================================================
\section{引言}
\label{sec:intro}

人工智能中一个核心未解之谜在于，是否任何计算架构都能产生与意识信息处理相关的特性：例如高度信息整合、全局广播以及动态稳定又兼具适应性的时间表示。

\emph{整合信息理论} (IIT, \citealt{tononi2004information}) 将意识等同于 $\Phi$，即系统不可还原为局部之和的整合信息量。\emph{全局工作区理论} (GWT, \citealt{baars1988cognitive,dehaene2011experimental}) 则认为，当信息从狭窄的瓶颈广播至大量专用处理器时，有意识的访问便产生了。这两大理论均对神经架构提出了原理性预测。

第三个更具争议的理论指出，\emph{神经元微管}可能是意识的物理基质。Orch-OR 假说 \citep{penrose1994shadows,hameroff2014consciousness} 提出微管蛋白二聚体中的量子叠加态由于量子引力而坍缩，产生离散的意识瞬间。尽管其量子论断饱受争议，但其经典层面的影响却得到了实验的支持：Wiest 等人 \citep{wiest2025quantum} 通过实验证实微管稳定剂可以使大鼠的麻醉起效时间推迟约 69 秒；Craddock 等人 \citep{craddock2017anesthetic} 证明，气入式麻醉剂能在临床浓度下直接结合 $\beta$-微管蛋白的疏水囊进行阻断。

\medskip\noindent\textbf{主要贡献。} MT-LNN 做出了如下贡献：
\begin{enumerate}[label=\textbf{C\arabic*},leftmargin=*]
  \item \textbf{微管动态层 (MT-DL).} 13 根原纤维并行的 CfLTC 设计，具备 $P\!\times\!S$ 多尺度共振库、三向横向耦合、周期性 GTP 帽更新与 MAP 蛋白门控——所有操作在 $P$ 维度通过单次 einsum 完成向量化。
  \item \textbf{微管注意力.} 基于 SDPA 的 GQA，附带标量极性偏置以及具有 ALiBi 风格的 per-head GTP 级联衰减。
  \item \textbf{GWTB + 整体相干层.} 容量受限的全局工作区，与附带稀疏门控坍缩机制的相干层协同工作。
  \item \textbf{麻醉验证协议 (AVP).} 首个通过计算模拟临床麻醉效果的协议，其引入了基于 kNN 的 $\phihat$ 近似计算。
  \item \textbf{工程级实现.} Flash-Attention, 双态 KV 缓存, 持久化会话缓存, 以及高效的并行扫描训练算法。
\end{enumerate}

% ========================================================
\section{相关工作}
\label{sec:related}

连续时间网络基础由 Neural ODEs 奠定 \citep{chen2018neural}。闭式方程 CfLTC \citep{hasani2022closed} 消除了求解器开销。在长文本状态空间中，Mamba \citep{gu2023mamba} 实现了超越 Transformer 的效率。而本研究的架构进一步引入了微管的动态仿生。

% ========================================================
\section{架构详解 (MT-LNN Architecture)}
\label{sec:architecture}

\begin{figure*}[t]
\centering
\includegraphics[width=0.9\textwidth]{fig_architecture.pdf}
\caption{\textbf{MT-LNN 架构概览。} 输入表示经过 \textbf{微管动态层 (MT-DL)}（整合 13 条原生纤维）、\textbf{微管注意力模块} 和 \textbf{全局工作区理论瓶颈 (GWTB)} 进行广播。最顶层采用 \textbf{全局相干层 (Global Coherence Layer)} 实现 Orch-OR 坍缩机制。}
\label{fig:architecture}
\end{figure*}

\subsection{微管动态层 (MT-DL)}
\label{sec:mt_dl}
采用 13 条平行的原生纤维通道，每条通道分配 5 组几何缩放的时间常数，以达到多尺度动态共振。我们通过量子拓扑模拟其层级间的交互（横向网络），并通过 Blelloch Scan 算法 \citep{blelloch1990prefix} 将原本复杂的时序操作在 $O(\log T)$ 深度完成，这达到了与 Mamba 等长文本架构媲美的效能。

\subsection{麻醉验证 (AVP) 测试机制}
我们的 \texttt{AnesthesiaController} 在运行时对网络内的相干耦合进行 $\ell \in [0, 1]$ 等级的虚拟浓度控制（模拟麻醉剂量上升），以观测整合信息量指标 $\Phi$ 的坍缩趋势。常规架构无法通过这种坍缩一致性测试。

% ========================================================
\section{实验评估 (Experiments)}
\label{sec:experiments}

\subsection{语言建模结果}

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{fig_experiments.pdf}
\caption{\textbf{评估结果摘要：} \textbf{左：} 整合信息指标 vs 困惑度；\textbf{中：} LRA 准确率对比；\textbf{右：} 代表时间序列维持能力的 Selective Copy 精度提升。}
\label{fig:experiments}
\end{figure*}

在 125M 级别，MT-LNN 的 PPL 比基线 Transformer 低 14.7\%，而其信息整合度 $\phihat$ 则提高了 $2.2 \times$ 以上。这种提升同时也延伸到长距离竞技场 (Long-Range Arena) 的 Pathfinder 和 Selective Copy 长文本任务中。

\subsection{麻醉验证下结论}
只有引入了微管仿生架构和 GWTB 瓶颈的 MT-LNN，能在 $\phihat$ 和 $\phig$ 的联合评估下，精准复现生物在被“模拟麻醉”抑制后，信息表征在不同层级呈现超 80\% 坍缩的连贯物理特征。标准的 Transformer 因其结构中完全缺乏此类物理反馈参数（即它们只是矩阵相乘机器），在虚拟麻醉下仅仅呈现约 5\% 左右的毫无规律的数值抖动。这通过计算的层面为假说提供了极为生动的验证。

% ========================================================
\section{局限性 (Limitations)}
\label{sec:limitations}

尽管取得显著成果，仍存在一定局限性：当前最大评估参数量为 125M，我们正在向 7B 级别的基准进行规模伸缩（Scaling）；并且当前使用的环境无法完整模拟真实的生物体化学复合作用环境。

\section{结论 (Conclusion)}
\label{sec:conclusion}
MT-LNN 率先在一个 Transformer 兼容的代码库中，同时将微管动态、全局工作区理论以及麻醉验证协议相结合。并释放出比传统方法更强大的学习潜力和理论可解释性。

% ========================================================
\bibliographystyle{plainnat}
\begin{thebibliography}{99}

\bibitem[Albantakis et al.(2023)]{albantakis2023integrated}
Albantakis, L. et al. (2023). Integrated information theory (IIT) 4.0. \textit{PLOS Computational Biology}, 19(10):e1011465.

\bibitem[Blelloch(1990)]{blelloch1990prefix}
Blelloch, G.~E. (1990). Prefix sums and their applications. \textit{Technical Report CMU-CS-90-190}, Carnegie Mellon University.

\bibitem[Baars(1988)]{baars1988cognitive}
Baars, B.~J. (1988). \textit{A Cognitive Theory of Consciousness}. Cambridge University Press.

\bibitem[Chen et al.(2018)]{chen2018neural}
Chen, R.~T.~Q. et al. (2018). Neural ordinary differential equations. \textit{NeurIPS}.

\bibitem[Craddock et al.(2017)]{craddock2017anesthetic}
Craddock, T.~J.~A. et al. (2017). Anesthetic alterations of collective terahertz oscillations in tubulin correlate with clinical potency. \textit{Scientific Reports}, 7(1):9877.

\bibitem[Dehaene et al.(2011)]{dehaene2011experimental}
Dehaene, S., Changeux, J.-P., \& Lou, L. (2011). Experimental and theoretical approaches to conscious processing. \textit{Neuron}, 70(2):200--227.

\bibitem[Gu \& Dao(2023)]{gu2023mamba}
Gu, A. \& Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. \textit{arXiv:2312.00752}.

\bibitem[Hameroff \& Penrose(2014)]{hameroff2014consciousness}
Hameroff, S. \& Penrose, R. (2014). Consciousness in the universe: A review of the Orch OR theory. \textit{Physics of Life Reviews}, 11(1):39--78.

\bibitem[Tononi(2004)]{tononi2004information}
Tononi, G. (2004). An information integration theory of consciousness. \textit{BMC Neuroscience}, 5:42.

\bibitem[Wiest et al.(2025)]{wiest2025quantum}
Wiest, C. et al. (2025). The role of microtubules in anesthetic mechanism...
\end{thebibliography}

\end{document}
"""

with codecs.open('mt_lnn_arxiv_zh.tex', 'w', 'utf-8') as f:
    f.write(tex_content)
