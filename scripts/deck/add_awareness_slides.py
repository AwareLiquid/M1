import os

file_path = "investor_deck_mt_lnn_zh.tex"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

target = r"""\begin{frame}{核心应用场景与竞争力分析}"""

injection = r"""\begin{frame}{终极产品模式：Awareness Network (端云混合智能网)}
\begin{itemize}
    \item \textbf{如何超越 Gemini 3.1 等万亿级云端模型？} 我们不拼大而全的云端百科全书，我们要做不对称的降维打击——用 \textbf{M1} 把巨无霸收编为外挂硬盘，而让 \textbf{M1} 成为系统的真正灵魂。
    \vspace{0.5em}
    \item \textbf{闭眼暗中推演 (Latent Reasoning Loop)：}
    \begin{itemize}
        \item 传统大模型只能靠疯狂打字（吐出废话 Token 作为草稿纸）来做表面思考。
        \item M1 的连续状态机在遇到难题时可以“静默十秒”，在底层用高维微分方程内部演算摩擦数十次。不用废话，直接给出终极真理，体验上更像深思熟虑的智者。
    \end{itemize}
    \vspace{0.3em}
    \item \textbf{一生在线的数字潜意识 (Personal State Capsule)：}
    \begin{itemize}
        \item Gemini 每一次对话都在“清空失忆”。
        \item M1 把用户极长生命周期内的几百万字操作记录，完美压缩在数十 KB 的 \textbf{个人状态胶囊 ($h_{prev}$)} 中。换台设备插入文件，新电脑瞬间继承与你长达三年的默契与直觉。
    \end{itemize}
    \vspace{0.3em}
    \item \textbf{白嫖云端算力，主导系统智力 (Cloud Oracle Router)：}
    \begin{itemize}
        \item 遇到缺乏事实资料时，M1 精准发送极简口令调取云大模型资料。在这里，强如 Gemini 3.1 也沦为了 Awareness Network 调用的一本“外部字典”，而拥有逻辑、直觉与隐私的 M1 才是决策与陪伴的不可替代者。
    \end{itemize}
\end{itemize}
\end{frame}

\begin{frame}{核心应用场景与竞争力分析}"""

if target in content and injection not in content:
    content = content.replace(target, injection)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Injected awareness slides successfully.")
else:
    print("Target not found or already injected.")
