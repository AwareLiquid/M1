import re

with open('e:/M1/investor_deck_mt_lnn_zh.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# Frame 1: MT-LNN 架构登场
new_f1 = r"""\begin{frame}{我们的方案：MT-LNN 架构登场}
\begin{columns}
\begin{column}{0.6\textwidth}
\textbf{预测编码与 $O(1)$ 微管液态系统：}
\begin{itemize}
    \item \textbf{重构前馈与缓存墙：} 抛弃标准 Transformer 中耗费显存的静态 FFN 和 $O(T)$ KV 缓存，替换为 13 个平行的多尺度连续时间网络以及 \textbf{$O(1)$ 工作记忆衰减矩阵}。
    \item \textbf{内置解释性与跳过机制：} 引入动态计算跳过（Endogenous Compute Skipping）指数级节省显存算力。内置因果链与自我监控提取头，打破大模型黑盒困境。
\end{itemize}
\end{column}
\begin{column}{0.4\textwidth}
\centering
\includegraphics[width=\textwidth]{fig_architecture.pdf}
\end{column}
\end{columns}
\end{frame}"""
text = re.sub(r'\\begin\{frame\}\{我们的方案：MT-LNN 架构登场\}[\s\S]*?\\end\{frame\}', new_f1, text)

# Frame 2: 超越“文字接龙”
new_f2 = r"""\begin{frame}{超越“文字接龙”：多尺度预测编码 (Predictive Coding)}
\begin{columns}
\begin{column}{0.55\textwidth}
\begin{itemize}
    \item \textbf{突破概率法则的局限：} 传统大模型本质上基于海量词汇的高概率路径进行盲目的“下一个词”拼接，并没有建立关于世界逻辑的深层推演。
    \item \textbf{原生跨尺度预测自我监督：} MT-LNN 实现了生物大脑标志性的\textbf{预测编码系统}。高层的抽象通道会自动自下而上地对底层的感知通道发送预测，并以此误差作为惩罚，迫使模型建立极强的深层物理因果模型。
\end{itemize}
\end{column}
\begin{column}{0.45\textwidth}
\centering
\includegraphics[width=0.85\textwidth]{logo.png}
\\ \vspace{0.5em} \textit{\footnotesize MT-LNN 核心动态定位概念}
\end{column}
\end{columns}
\end{frame}"""
text = re.sub(r'\\begin\{frame\}\{超越“文字接龙”：真正理解真实物理世界规律\}[\s\S]*?\\end\{frame\}', new_f2, text)

# Frame 3: 深度解析
new_f3 = r"""\begin{frame}{深度解析：$O(1)$ 工作记忆与算力跳过}
\begin{itemize}
    \item \textbf{打破注意力上下文瓶颈：} MT-LNN 引入指数衰减的全局工作记忆（Decay Working Memory）。新 Token 的整合开销为极其严格的常量 $O(1)$，彻底消灭了 GPU 推理时的内存刺客。
    \item \textbf{动态算力跳过 (Compute Skipping)：} 仿生大脑，在面对高度冗余或简单的上下文区间时，休眠的原丝通道会被原生覆盖掩码跳过计算（Masked out），直接实现云计算算力的指数级成本节约。
    \item \textbf{量子启发耦合：} 创新性引入类似量子物理的隐状态交互，使不同频段通道之间在时间流中实现信息平稳交换过渡。
\end{itemize}
\end{frame}"""
text = re.sub(r'\\begin\{frame\}\{深度解析：并行扫描与量子耦合\}[\s\S]*?\\end\{frame\}', new_f3, text)

with open('e:/M1/investor_deck_mt_lnn_zh.tex', 'w', encoding='utf-8') as f:
    f.write(text)
