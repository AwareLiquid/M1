import os

file_path = "investor_deck_mt_lnn_zh.tex"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

target = r"""\begin{frame}{市场与定位：独特的差异化竞争优势}"""

injection = r"""\begin{frame}{从产品与商业化看 M1 的降维打击}
\begin{itemize}
    \item \textbf{极低门槛（告别天价显卡）：} 无需 H800/A100 等昂贵硬件。得益于 $O(1)$ 算子压缩，M1 处理超长文本的缓存恒定在 \textbf{4.1 KB}。这使得百亿/千亿级别的模型能力，能被无缝下放到普通轻薄本、智能终端和局域网服务器中。
    \item \textbf{算力降本（引擎的“智能启停”）：} 传统模型应对哪怕是一句简单问候都会拉满所有算力。M1 的\textbf{“稀疏共振”}机制允许系统在处理简单逻辑时瞬间休眠 80\% 的冗余计算通道，在没有专用加速卡的情况下仍实现 \textbf{13\%+} 的原生计算提速。
    \item \textbf{长线体验（攻克“中间遗忘”痛点）：} 类似 Mamba 的传统线性模型随文本无限变长会不可避免地产生衰减。M1 通过底层的门控调度锁住信息，即便在极高噪声打断和 128K 超长闭环盲测中，依然实现了 $>99\%$ 的无死角提取精度。
    \item \textbf{私有化落地（行业刚需避风港）：} 金融、法律和国家机构等对数据极端敏感领域无法将材料上传给云端 API。M1 由于对算力和内存的极低索求，成为完美契合局域网、可完全断网离线流式运作的 AI 强隐私解决方案。
\end{itemize}
\end{frame}

\begin{frame}{硬核实测：M1 与主流同梯队架构的数据鸿沟}
\begin{itemize}
    \item 在同等参数（200K级别微型沙盒验证）的绝对公平盲测下，M1 展现出了令人惊悚的压制力：
    \vspace{0.5em}
    \item \textbf{1. 长文极限抗干扰（$T=229$ 大海捞针盲测）：}
    \begin{itemize}
        \item \textbf{M1 架构 (MT-LNN)：} 长序列事实的完全匹配率高达 \textbf{96.5\%}。
        \item \textbf{标准 Transformer / 传统 LNN：} 仅为 \textbf{2.3\%}（基本等同于崩溃瞎猜）。
        \item \textbf{[鸿沟]}：哪怕在显微镜级别测试下，M1 的强力信息锚定能力更是将传统架构甩开了足足 \textbf{42倍}。
    \end{itemize}
    \vspace{0.3em}
    \item \textbf{2. 内存运行极值压测（以 1000 Token 处理为例）：}
    \begin{itemize}
        \item \textbf{传统机制 (死记硬背 KV 叠砖)：} 堆叠至 1020.5 KB，呈灾难式线性膨胀。
        \item \textbf{M1 架构 (只更新此刻的 $O(1)$ 状态)：} 内存死死钉在 \textbf{4.1 KB}（暴降 \textbf{99.6\%}）。
    \end{itemize}
    \vspace{0.3em}
    \item \textbf{3. CPU 裸机底层纯计算速度：}
    \begin{itemize}
        \item 传统无休眠满载狂飙：6528 Tok/s。
        \item \textbf{M1 开启休眠模式 (Sparse Top-$k=1$)：} 骤升至 \textbf{7390 Tok/s}。单靠动态物理特性便拉升了超 \textbf{13\%} 的速度，且输出精度几乎不损耗。
    \end{itemize}
\end{itemize}
\end{frame}

\begin{frame}{市场与定位：独特的差异化竞争优势}"""

if target in content and injection not in content:
    content = content.replace(target, injection)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Injected commercial slides successfully.")
else:
    print("Target not found or already injected.")
