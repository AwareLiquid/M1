# -*- coding: utf-8 -*-
import re

def update_fonts(text):
    if r'\newfontfamily\aeonik{Aeonik}' not in text:
        text = text.replace(r'\usepackage{tikz}', r'\usepackage{tikz}' + '\n\\usepackage{fontspec}\n\\newfontfamily\\aeonik{Aeonik}')
    
    old_footline = r'''\begin{beamercolorbox}[wd=.333333\paperwidth,ht=2.25ex,dp=1ex,left]{author in head/foot}%
    \hspace*{2ex}\textbf{Awareness}
  \end{beamercolorbox}%
  \begin{beamercolorbox}[wd=.333333\paperwidth,ht=2.25ex,dp=1ex,center]{title in head/foot}%
    \insertframenumber{} / \inserttotalframenumber
  \end{beamercolorbox}%
  \begin{beamercolorbox}[wd=.333333\paperwidth,ht=2.25ex,dp=1ex,right]{date in head/foot}%
    \textbf{Confidential \& Proprietary}\hspace*{2ex}
  \end{beamercolorbox}'''
    
    new_footline = r'''\begin{beamercolorbox}[wd=.333333\paperwidth,ht=2.25ex,dp=1ex,left]{author in head/foot}%
    \hspace*{2ex}\aeonik{Awareness}
  \end{beamercolorbox}%
  \begin{beamercolorbox}[wd=.333333\paperwidth,ht=2.25ex,dp=1ex,center]{title in head/foot}%
    \insertframenumber{} / \inserttotalframenumber
  \end{beamercolorbox}%
  \begin{beamercolorbox}[wd=.333333\paperwidth,ht=2.25ex,dp=1ex,right]{date in head/foot}%
    \aeonik{Confidential \& Proprietary}\hspace*{2ex}
  \end{beamercolorbox}'''
    
    text = text.replace(old_footline, new_footline)
    return text

def add_slides_en(text):
    schools_slide = r'''\begin{frame}{AI Industry's 8 Core Schools}
\begin{itemize}
    \item \textbf{1. Symbolism:} Logical and visual reasoning, closest to human conscious thought.
    \item \textbf{2. Bayesianism:} Prior/posterior probability reasoning via Bayesian networks.
    \item \textbf{3. Analogizer:} SVMs and nearest-neighbor classification.
    \item \textbf{4. Evolutionism:} Mutational and evolutionary algorithms.
    \item \textbf{5. Connectionism (Current Mainstream):} Neural networks, Large Models, XAI, and \textbf{Liquid Neural Networks (LNN)}.
    \item \textbf{6. Reinforcement Learning:} Reward-punishment optimization.
    \item \textbf{7. Multi-Agent Systems:} Simulating interaction between multiple intelligent agents.
    \item \textbf{8. Continual Learning:} AI evolving continuously without total retraining.
\end{itemize}
\end{frame}
'''
    charts_slide = r'''\begin{frame}{Market Positioning \& Concurrency Competitiveness}
\begin{columns}
\begin{column}{0.5\textwidth}
\centering
\includegraphics[width=\textwidth]{concurrency.png}
\end{column}
\begin{column}{0.5\textwidth}
\centering
\includegraphics[width=\textwidth]{landscape.png}
\end{column}
\end{columns}
\end{frame}
'''
    if "8 Core Schools" not in text:
        text = text.replace(r'\begin{frame}{1. Executive Summary}', schools_slide + '\n' + r'\begin{frame}{1. Executive Summary}')
    
    if "Market Positioning \& Concurrency" not in text:
        match = re.search(r'\\begin\{frame\}\{\d+\.\s*Competitive Landscape Matrix.*?\}.*?\\end\{frame\}', text, flags=re.DOTALL)
        if match:
            text = text.replace(match.group(0), match.group(0) + '\n\n' + charts_slide)
    
    return text

def add_slides_zh(text):
    schools_slide = r'''\begin{frame}{一、AI八大核心流派}
\begin{itemize}
    \item \textbf{1. 符号主义：}靠逻辑运算、视觉推理，最贴近人类思维。
    \item \textbf{2. 贝叶斯主义：}用先验/后验概率推理，核心是贝叶斯网络。
    \item \textbf{3. 类比主义：}靠事物类比做分类，代表SVM、近邻算法。
    \item \textbf{4. 演化主义：}模拟自然演化规律设计AI。
    \item \textbf{5. 联结主义（当前主流）：}神经网络技术，包含大模型、可解释AI（XAI）、\textbf{液态神经网络 (LNN)}。
    \item \textbf{6. 强化学习：}模拟生物“奖励-惩罚”机制训练AI。
    \item \textbf{7. 多代理系统：}模拟多智能体之间的交互与决策。
    \item \textbf{8. 连续学习：}AI无需重新训练，就能持续学习新内容。
\end{itemize}
\end{frame}
'''
    charts_slide = r'''\begin{frame}{市场定位与高并发竞争象限}
\begin{columns}
\begin{column}{0.5\textwidth}
\centering
\includegraphics[width=\textwidth]{concurrency.png}
\end{column}
\begin{column}{0.5\textwidth}
\centering
\includegraphics[width=\textwidth]{landscape.png}
\end{column}
\end{columns}
\end{frame}
'''
    if "AI八大核心流派" not in text:
        text = text.replace(r'\begin{frame}{1. AI 行业的内存危机}', schools_slide + '\n' + r'\begin{frame}{1. AI 行业的内存危机}')
    
    if "市场定位与高并发竞争象限" not in text:
        match = re.search(r'\\begin\{frame\}\{\d+\.\s*生态竞争.*?\}.*?\\end\{frame\}', text, flags=re.DOTALL)
        if match:
            text = text.replace(match.group(0), match.group(0) + '\n\n' + charts_slide)
            
    return text

def renumber(text):
    frames = re.findall(r'\\begin\{frame\}\{.*?\}', text)
    counter = 1
    for f in frames:
        if 'Contact' in f or 'Appendix' in f or '联系' in f or '图表' in f:
            pass
        else:
            new_f = re.sub(r'\\begin\{frame\}\{[^A-Za-z0-9]*\d*\.?\s*', f'\\\\begin{{frame}}{{{counter}. ', f)
            if new_f != f:
                text = text.replace(f, new_f, 1)
            counter += 1
    return text

# Process English
with open('investor_deck_mt_lnn.tex', 'r', encoding='utf-8') as f:
    en_text = f.read()
en_text = update_fonts(en_text)
en_text = add_slides_en(en_text)
en_text = renumber(en_text)
with open('investor_deck_mt_lnn.tex', 'w', encoding='utf-8') as f:
    f.write(en_text)

# Process Chinese
with open('investor_deck_mt_lnn_zh.tex', 'r', encoding='utf-8') as f:
    zh_text = f.read()
zh_text = update_fonts(zh_text)
zh_text = add_slides_zh(zh_text)
zh_text = renumber(zh_text)
with open('investor_deck_mt_lnn_zh.tex', 'w', encoding='utf-8') as f:
    f.write(zh_text)

print("Updates applied")
