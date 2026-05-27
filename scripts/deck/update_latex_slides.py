import re
import matplotlib.pyplot as plt

def process_file(filename, slide_content, split_pattern, new_split_pattern, max_frames):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    header_footer_tex = r"""
\usepackage{tikz}

\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{footline}{
  \leavevmode%
  \hbox{%
  \begin{beamercolorbox}[wd=.333333\paperwidth,ht=2.25ex,dp=1ex,left]{author in head/foot}%
    \hspace*{2ex}\textbf{Awareness}
  \end{beamercolorbox}%
  \begin{beamercolorbox}[wd=.333333\paperwidth,ht=2.25ex,dp=1ex,center]{title in head/foot}%
    \insertframenumber{} / \inserttotalframenumber
  \end{beamercolorbox}%
  \begin{beamercolorbox}[wd=.333333\paperwidth,ht=2.25ex,dp=1ex,right]{date in head/foot}%
    \textbf{Confidential \& Proprietary}\hspace*{2ex}
  \end{beamercolorbox}}%
  \vskip0pt%
}

\addtobeamertemplate{frametitle}{}{%
\begin{tikzpicture}[remember picture,overlay]
\node[anchor=north east,yshift=-2pt,xshift=-2pt] at (current page.north east) {\includegraphics[height=1.2cm]{logo.png}};
\end{tikzpicture}}

\begin{document}
"""

    content = content.replace(r'\begin{document}', header_footer_tex)

    parts = content.split(split_pattern)
    if len(parts) == 2:
        content = parts[0] + slide_content + new_split_pattern + parts[1]
        for i in range(max_frames, 5, -1):
            content = re.sub(rf'\\begin{{frame}}\{{{i}\.', rf'\\begin{{frame}}{{{i+1}.', content)
            
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)

en_slide = r"""
\begin{frame}{6. Beyond Next-Word Prediction: Understanding the World}
\begin{columns}
\begin{column}{0.55\textwidth}
\begin{itemize}
    \item \textbf{Transformer's Illusion:} Standard models can perfectly predict the sequence: \textit{``The apple fell from the tree to the ground.''} However, they only rely on statistical word correlations. They do not understand the physical process or the concept of ``gravity''.
    \item \textbf{Can MT-LNN understand it? Yes.}
    \item \textbf{How?} MT-LNN abandons pure static statistics. Its continuous-time liquid differential equations form a dynamic causal model. It maps the state of the ``apple'' moving over ``time'', simulating the underlying physical rules (gravity) to reason about what happens next.
\end{itemize}
\end{column}
\begin{column}{0.45\textwidth}
\centering
\includegraphics[width=0.85\textwidth]{logo.png}
\\ \vspace{0.5em} \textit{\footnotesize MT-LNN Dynamic Concept Positioning}
\end{column}
\end{columns}
\end{frame}
"""

zh_slide = r"""
\begin{frame}{6. 超越“文字接龙”：真正理解世界模型}
\begin{columns}
\begin{column}{0.55\textwidth}
\begin{itemize}
    \item \textbf{知其然而不知其所以然：} 传统大模型（Transformer）可以完美预测出语句 \textit{“苹果从树上掉到了地上”}。但它仅仅是在做概率最高词汇的拼凑，完全无法理解真实世界发生的“物理过程”，更不懂什么是“重力”。
    \item \textbf{MT-LNN 能做到真正理解吗？可以！}
    \item \textbf{我们是如何做到的？} 完全放弃纯粹的静态词频矩阵。MT-LNN 独特的连续时间微分方程具有“流体动态感知”。它在内部隐状态中，真正构建了“苹果”随着“时间”下落的因果世界模型。不靠死记硬背，而是靠实打实地推演物理法则！
\end{itemize}
\end{column}
\begin{column}{0.45\textwidth}
\centering
\includegraphics[width=0.85\textwidth]{logo.png}
\\ \vspace{0.5em} \textit{\footnotesize MT-LNN 核心动态定位概念}
\end{column}
\end{columns}
\end{frame}
"""

process_file('investor_deck_mt_lnn.tex', en_slide, r'\begin{frame}{6. The MT-LNN Architecture}', r'\begin{frame}{7. The MT-LNN Architecture}', 25)
process_file('investor_deck_mt_lnn_zh.tex', zh_slide, r'\begin{frame}{6. 深度解析：并行扫描与量子耦合}', r'\begin{frame}{7. 深度解析：并行扫描与量子耦合}', 20)

fig, ax = plt.subplots(figsize=(2,2))
circle1 = plt.Circle((0.5, 0.5), 0.4, color='#a0b0c0', fill=False, linewidth=5)
circle2 = plt.Circle((0.5, 0.5), 0.2, color='#a0b0c0', fill=False, linewidth=5)
circle3 = plt.Circle((0.35, 0.75), 0.1, color='#a0b0c0', fill=False, linewidth=3)
circle4 = plt.Circle((0.35, 0.75), 0.03, color='#a0b0c0')
ax.add_patch(circle1)
ax.add_patch(circle2)
ax.add_patch(circle3)
ax.add_patch(circle4)
ax.axis('off')
plt.savefig('logo.png', transparent=True, dpi=300)

print("Modification scripts completed successfully.")