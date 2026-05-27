import re

with open('e:/M1/investor_deck_mt_lnn_zh.tex', 'r', encoding='utf-8') as f:
    text = f.read()

def inject(pattern, replacement, string):
    match = re.search(pattern, string)
    if not match: return string
    return string[:match.start()] + replacement + string[match.end():]

# We will inject a new frame addressing "Operator Compression & Sparse Resonance" before "技术护城河"
new_frame = r"""
\begin{frame}{核心实现：算子压缩与稀疏共振}
\begin{columns}
\begin{column}{0.5\textwidth}
\textbf{状态流式算子压缩 (Operator Compression):}
\begin{itemize}
    \item 摒弃传统 $O(T)$ 的庞大 KV 缓存，在流式推理时将上下文仅保留为循环隐状态 $h_{prev}$。
    \item \textbf{测试数据：} 在 1000 Tokens 时，传统 KV 需要 1020 KB，而 M1 O(1) 状态模式仅需恒定的 \textbf{4.1 KB}。
\end{itemize}

\vspace{0.5em}
\textbf{动态稀疏共振 (Sparse Resonance):}
\begin{itemize}
    \item 利用 Top-$k$ 门控剔除沉寂时间尺度（保留 2/5），跳过冗余密度计算。
    \item \textbf{加速效果：} CPU 环境下推理每秒 Token 数从 3650 飙升至 \textbf{6400+}。
\end{itemize}
\end{column}
\begin{column}{0.5\textwidth}
\centering
\includegraphics[width=\textwidth]{fig_operator_compression_updates.pdf}
\end{column}
\end{columns}
\end{frame}

\begin{frame}{技术护城河与壁垒}"""

# Inject before 技术护城河
text = inject(r'\\begin\{frame\}\{技术护城河与壁垒\}', new_frame, text)

with open('e:/M1/investor_deck_mt_lnn_zh.tex', 'w', encoding='utf-8') as f:
    f.write(text)
