# -*- coding: utf-8 -*-
import re

with open('investor_deck_mt_lnn_zh.tex', 'r', encoding='utf-8') as f:
    text = f.read()

zh_team_ask = r'''\begin{frame}{融资方案与阶段目标 (The Ask)}
\begin{columns}
\begin{column}{0.5\textwidth}
\textbf{本轮诉求：Seed / A 轮 - \.5M}
\vspace{0.5em}
\begin{itemize}
    \item 火速上马 7B 级原生大模型基座炼制。
    \item 组建销售铁军，抢占 B 端超级红利。
    \item \textbf{核心目标：} 12 个月内斩获 10 家全球头部企业 PoC，锁定 \.5M 年度经常性营收 (ARR)。
\end{itemize}
\end{column}
\begin{column}{0.5\textwidth}
\textbf{资金使用占比：}
\begin{itemize}
    \item \textbf{50\% 算力与研发：} 租赁大规模 H100 集群突破预训练。
    \item \textbf{30\% 架构工程师体系：} 重金吸纳顶尖 CUDA 与系统底层算子极致优化极客。
    \item \textbf{20\% 市场与 GTM：} B 端大客户节点强势攻坚。
\end{itemize}
\end{column}
\end{columns}
\end{frame}

\begin{frame}{核心团队：Awareness}
\begin{itemize}
    \item \textbf{Everest An -- 创始人} (AI 记忆数据库资深极客)
    \vspace{0.5em}
    \item \textbf{去中心化基建专家：} 前 IPFS Athna 核心成员，曾主导并全面负责中国地区企业级数据库的底层建设。
    \item \textbf{头部资本与生态视角：} 曾任出海顶级基金 Outliers Fund 投资经理。主导并参投顶级公链体系 (Dfinity、Harmony)、前沿基础设施 (Analog)，以及区块链核心代码安全审计机构 (Quantstamp)。
    \item \textbf{世界级极客硬实力：} 驰呈技术前沿，在竞争最激烈的全球以太坊黑客马拉松大赛中多次斩获硬核冠军。
\end{itemize}
\end{frame}
'''

contact_slide = r'''
\begin{frame}{联系我们}
\begin{center}
    \LARGE \textbf{Awareness} \\
    \vspace{0.5em}
    \large Everest An -- 创始人 / 核心架构师 \\
    \vspace{1.5em}
    \normalsize
    \textbf{Email:} everest9812@gmail.com \\
    \textbf{WeChat / Telegram:} EverestAn \\
    \vspace{1.5em}
    \textit{打碎旧规则，共同开启液态记忆计算的黄金时代。}
\end{center}
\end{frame}

\end{document}
'''

# insert before 18
text = text.replace(r'\begin{frame}{18. 我们坚如磐石的技术防卫高墙}', zh_team_ask + '\n' + r'\begin{frame}{18. 我们坚如磐石的技术防卫高墙}')

# replace end document
text = text.replace(r'\end{document}', contact_slide)

# renumber frames sequentially
frames = re.findall(r'\\begin\{frame\}\{.*?\}', text)
counter = 1
for f in frames:
    new_f = re.sub(r'\{\d+\.\s*', f'{{{counter}. ', f)
    if new_f != f:
        text = text.replace(f, new_f, 1) # only replace first occurrence
    counter += 1

with open('investor_deck_mt_lnn_zh.tex', 'w', encoding='utf-8') as f:
    f.write(text)

print("done zh")
