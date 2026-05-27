import re

def fix_table_overflow(content):
    # Wrap any \begin{tabular}...\end{tabular} in a \resizebox{\textwidth}{!}{...} if not already wrapped
    if r'\resizebox' not in content and r'\begin{tabular}' in content:
        content = re.sub(r'(\\begin\{tabular\}.*?\\end\{tabular\})', r'\\resizebox{\\textwidth}{!}{\1}', content, flags=re.DOTALL)
    return content

def add_contact_en(content):
    contact_slide = r'''
\begin{frame}{Contact Us}
\begin{center}
    \LARGE \textbf{Awareness} \\
    \vspace{0.5em}
    \large Everest An - Founder \\
    \vspace{1.5em}
    \normalsize
    \textbf{Email:} everest9812@gmail.com \\
    \textbf{WeChat / Telegram:} EverestAn \\
    \vspace{1em}
    \textit{Join the next wave of Liquid AI Memory.}
\end{center}
\end{frame}
'''
    if 'everest9812@gmail.com' not in content:
        content = content.replace(r'\end{document}', contact_slide + '\n\\end{document}')
    return content

def add_contact_zh(content):
    contact_slide = r'''
\begin{frame}{联系我们 / Contact Information}
\begin{center}
    \LARGE \textbf{Awareness} \\
    \vspace{0.5em}
    \large Everest An - 创始人 (Founder) \\
    \vspace{1.5em}
    \normalsize
    \textbf{Email:} everest9812@gmail.com \\
    \textbf{WeChat / TG:} EverestAn \\
    \vspace{1em}
    \textit{欢迎交流，共同探索下一代液态算力革命。}
\end{center}
\end{frame}
'''
    if 'everest9812@gmail.com' not in content:
        content = content.replace(r'\end{document}', contact_slide + '\n\\end{document}')
    return content

def update_english():
    with open('investor_deck_mt_lnn.tex', 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Fix Table overflow
    content = fix_table_overflow(content)

    # 2. Update Team Slide
    old_team = re.search(r'\\begin\{frame\}\{\d+\.\s*Team.*?\}.*?\\end\{frame\}', content, flags=re.DOTALL)
    if old_team:
        new_team = r'''\begin{frame}{Team: Awareness}
\begin{itemize}
    \item \textbf{Everest An -- Founder (AI Memory Database)}
    \vspace{0.5em}
    \item \textbf{Decentralized Infra \& Cryptography Expert:} Former member of IPFS Athna, leading enterprise database construction in the Greater China region.
    \item \textbf{Investment \& Audit Background:} Former Investment Manager at Outliers Fund. Led ecosystem investments spanning L1 Public Chains (Dfinity, Harmony), Infrastructure (Analog), and Smart Contract Auditing (Quantstamp).
    \item \textbf{Engineering Excellence:} Multiple-time Champion at top-tier Global Ethereum Hackathons.
\end{itemize}
\vspace{1em}
\textit{Deeply combining heavy-duty blockchain cryptography consensus logic directly into the structural core of memory limits.}
\end{frame}'''
        content = content.replace(old_team.group(0), new_team)

    # 3. Enhance Asks Slide to include specific target milestones as requested by BP logic
    ask_match = re.search(r'\\begin\{frame\}\{\d+\.\s*The Ask.*?\}.*?\\end\{frame\}', content, flags=re.DOTALL)
    if ask_match and '10 Enterprise PoCs' not in ask_match.group(0):
        new_ask = ask_match.group(0).replace(r'\end{itemize}', r'\item \textbf{Immediate Objective:} Reach \.5M ARR \& 10 successful Enterprise PoCs within 12 months.' + '\n\\end{itemize}', 1)
        content = content.replace(ask_match.group(0), new_ask)

    # 4. Add Contact Slide
    content = add_contact_en(content)

    # 5. Fix frame numbers (cleanup everything to sequence properly)
    frames = re.findall(r'\\begin\{frame\}\{.*?\}', content)
    counter = 1
    for f in frames:
        if re.search(r'\{\d+\.', f):
            new_f = re.sub(r'\{\d+\.', f'{{{counter}.', f)
            content = content.replace(f, new_f, 1)
            counter += 1

    with open('investor_deck_mt_lnn.tex', 'w', encoding='utf-8') as f:
        f.write(content)

def update_chinese():
    with open('investor_deck_mt_lnn_zh.tex', 'r', encoding='utf-8') as f:
        content = f.read()

    content = fix_table_overflow(content)

    # Missing slides in Chinese: Need Team and Ask
    zh_team_ask = r'''
\begin{frame}{团队阵容 / Core Team: Awareness}
\begin{itemize}
    \item \textbf{Everest An -- 创始人} (AI 记忆数据库主架构师)
    \vspace{0.5em}
    \item \textbf{密码学与去中心化基建专家：} 前 IPFS Athna 核心成员，曾全面负责中国地区企业级数据库的建设与突围。
    \item \textbf{跨域顶层视角：} 曾任 Outliers Fund 投资经理。深度布局与投资了公链系统 (Dfinity、Harmony)、底层基础设施 (Analog)，以及顶级代码审计安全 (Quantstamp)。
    \item \textbf{极客极客实力：} 在竞争最激烈的全球以太坊黑客马拉松大赛中多次斩获冠军。
\end{itemize}
\end{frame}

\begin{frame}{融资方案与核心阶段目标}
\begin{columns}
\begin{column}{0.5\textwidth}
\textbf{本轮诉求：Seed / A 轮 - \.5M}
\begin{itemize}
    \item 火速上马 7B 级原生商用大模型炼制。
    \item 组建专项 B 端地推铁军，斩出首批高价值营收。
\end{itemize}
\vspace{1em}
\textbf{预计达成里程碑 (未来12个月)：}
\begin{itemize}
    \item 签下 10 家头部律所与金融资管机构完成 PoC (概念验证) 测试。
    \item 锁定至少 150 万美金年度经常性营收 (ARR)。
\end{itemize}
\end{column}
\begin{column}{0.5\textwidth}
\textbf{资金使用占比：}
\begin{itemize}
    \item \textbf{50\% 算力与研发：} 租赁购置 H100 级别的加速器阵列，推进千亿级数据清洗。
    \item \textbf{30\% 架构工程师体系：} 重金吸纳顶尖 CUDA 与算子优化极客。
    \item \textbf{20\% 市场与 GTM：} 直接攻坚 B 端大客户节点。
\end{itemize}
\end{column}
\end{columns}
\end{frame}
'''
    if 'Everest An -- 创始人' not in content:
        # Insert them before roadmap or before conclusion
        if r'\begin{frame}{18. 我们坚如磐石的技术防卫高墙}' in content:
            content = content.replace(r'\begin{frame}{18. 我们坚如磐石的技术防卫高墙}', zh_team_ask + '\n' + r'\begin{frame}{18. 我们坚如磐石的技术防卫高墙}')
        elif r'\begin{frame}{19. 尾声与结语总结}' in content:
             content = content.replace(r'\begin{frame}{19. 尾声与结语总结}', zh_team_ask + '\n' + r'\begin{frame}{19. 尾声与结语总结}')
        else:
             content = content.replace(r'\end{document}', zh_team_ask + '\n\\end{document}')

    # Add Contact Slide
    content = add_contact_zh(content)

    # Renumbering
    frames = re.findall(r'\\begin\{frame\}\{.*?\}', content)
    counter = 1
    for f in frames:
        if re.search(r'\{\d+\.', f):
            new_f = re.sub(r'\{\d+\.', f'{{{counter}.', f)
            content = content.replace(f, new_f, 1)
            counter += 1

    with open('investor_deck_mt_lnn_zh.tex', 'w', encoding='utf-8') as f:
        f.write(content)

update_english()
update_chinese()
print("Success")
