import os

# --- CHINESE DECK ---
zh_file = "investor_deck_mt_lnn_zh.tex"
with open(zh_file, "r", encoding="utf-8") as f:
    zh_content = f.read()

zh_target_old = r"""\begin{frame}{终极产品模式：Awareness Network (端云混合智能网)}
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
\end{frame}"""

zh_injection = r"""\begin{frame}
  \begin{center}
      \vspace{1.5em}
      {\Huge \textbf{终极商业形态：Awareness Network}} \\
      \vspace{1em}
      {\Large (端云混合边缘智能网)}
  \end{center}
\end{frame}

\begin{frame}{降维打击：从“云端内卷”到“端侧灵魂”}
\begin{itemize}
    \item \textbf{战略破局：} 放弃与 Gemini 3.1 等万亿级云端巨无霸在“百科全书广度”上烧钱堆算力，采取不对称的边缘降维打击。
    \item \textbf{核心定位：} 用 \textbf{M1} 把巨无霸收编为“无感情的外置硬盘”，而让本地的 \textbf{M1} 成为整个系统的真正灵魂核心。
    \item \textbf{架构形态：} Awareness Network 提出了“端云分离、状态主导”的混合边缘智能系统（Hybrid Edge-State Intelligence System）。
    \item \textbf{数据主权重构：} 核心逻辑与个人隐私永远流转于本地脱网设备的 4.1 KB $O(1)$ 状态胶囊中，彻底切断厂商“收割数据与按次收税”的做法。
\end{itemize}
\end{frame}

\begin{frame}{Awareness Network 架构示意图}
  \begin{figure}
    \centering
    \includegraphics[width=\textwidth]{fig_awareness_network}
  \end{figure}
\end{frame}

\begin{frame}{四大核心功能模块的产品体验降维}
\begin{itemize}
    \item \textbf{1. 闭眼暗中推演 (Latent Reasoning Loop)：}
    \begin{itemize}
        \item \textbf{云端巨头：} 靠疯狂吐出废话 Token 作为草稿纸（CoT）进行表面模型假思考。
        \item \textbf{M1 架构：} 遇到难题通过隐层微分方程循环几十次“静默计算”。无冗长输出直接洞穿真理，呈现类人智者体验。
    \end{itemize}
    \vspace{0.3em}
    \item \textbf{2. 一生在线的潜意识 (Personal State Capsule)：}
    \begin{itemize}
        \item \textbf{云端巨头：} 每日对话失忆，需通过极度耗资的外部 RAG 去找回记忆。
        \item \textbf{M1 架构：} 把百万字交互压缩为恒定极小的 $h_{prev}$ 胶囊。换新设备导入即刻继承长达三年的直觉与潜意识。
    \end{itemize}
    \vspace{0.3em}
    \item \textbf{3. 主动预测纠错 (Predictive Error Monitor)：}
    \begin{itemize}
        \item M1 利用顶层主动向下预测用户意图，一旦逻辑背离，立刻由“被动回答”变为“主动亮红灯截断”护航。
    \end{itemize}
    \vspace{0.3em}
    \item \textbf{4. 云端打工节点 (Cloud Oracle Router)：}
    \begin{itemize}
        \item 遭遇常识盲区时，M1 自动发送极简口令调取云中心大模型。Gemini 被贬为干苦力的资料“供货商”，最终经 M1 以极其私人化的同理心语境加工回答。
    \end{itemize}
\end{itemize}
\end{frame}"""

if zh_target_old in zh_content:
    zh_content = zh_content.replace(zh_target_old, zh_injection)
    with open(zh_file, "w", encoding="utf-8") as f:
        f.write(zh_content)
    print("M1 zh deck updated.")
else:
    print("Cannot find old slide in ZH deck")

# --- ENGLISH DECK ---
en_file = "investor_deck_mt_lnn.tex"
with open(en_file, "r", encoding="utf-8") as f:
    en_content = f.read()

en_target = r"""\begin{frame}{Market Sizing (TAM/SAM/SOM)}"""

en_injection = r"""\begin{frame}
  \begin{center}
      \vspace{1.5em}
      {\Huge \textbf{The Ultimate Vision: Awareness Network}} \\
      \vspace{1em}
      {\Large (Hybrid Edge-State Intelligence System)}
  \end{center}
\end{frame}

\begin{frame}{Asymmetric Warfare vs. The Cloud Paradigm}
\begin{itemize}
    \item \textbf{Strategic Disruption:} Rather than burning capital to compete with trillion-parameter behemoths like Gemini 3.1 & GPT-4o on "encyclopedic breadth", M1 enforces an asymmetric drop in dimensionality.
    \item \textbf{Core Philosophy:} Subjugate cloud behemoths into "emotionless external hard drives", establishing the local \textbf{M1 Engine} as the true continuous "soul" of the intelligent user interface.
    \item \textbf{Hybrid Edge-State Architecture:} The Awareness Network runs purely local logic, routing to the cloud only for factual retrieval while maintaining absolute data sovereignty.
    \item \textbf{Privacy by Physics:} Core cognitive state remains perpetually contained within a local 4.1 KB $O(1)$ State Capsule, completely severing the "harvest and charge" cycle of cloud providers.
\end{itemize}
\end{frame}

\begin{frame}{Awareness Network Architecture}
  \begin{figure}
    \centering
    \includegraphics[width=\textwidth]{fig_awareness_network}
  \end{figure}
\end{frame}

\begin{frame}{Four Pillars of the Experiential Shift}
\begin{itemize}
    \item \textbf{1. Latent Reasoning Loop (Silent Deep Deduction):}
    \begin{itemize}
        \item \textbf{Cloud AI:} Fakes reasoning by spitting out verbose Chain-of-Thought (CoT) tokens.
        \item \textbf{M1:} Embarks on 10 seconds of "silent calculation" running internal ODEs natively. It strikes truth without bloated text, exhibiting wise, human-like intuition.
    \end{itemize}
    \vspace{0.3em}
    \item \textbf{2. Lifelong Personal State Capsule:}
    \begin{itemize}
        \item \textbf{Cloud AI:} Amnesic per session, requiring external Vector RAG.
        \item \textbf{M1:} Compresses millions of inputs into a constant $h_{prev}$. Load the capsule onto a new device, and it instantly inherits years of synchronized synergy.
    \end{itemize}
    \vspace{0.3em}
    \item \textbf{3. Predictive Error Monitor (Active AI):}
    \begin{itemize}
        \item Shifts from passive Q\&A to active prediction. It anticipates intent top-down and triggers red-light interventions the moment user logic violates globally established context.
    \end{itemize}
    \vspace{0.3em}
    \item \textbf{4. Cloud Oracle Router:}
    \begin{itemize}
        \item In factual blind spots, M1 dispatches surgical queries to cloud APIs. Gemini is demoted to a mere "data supplier," with M1 synthesizing the facts through its highly personalized state.
    \end{itemize}
\end{itemize}
\end{frame}

\begin{frame}{Market Sizing (TAM/SAM/SOM)}"""

if en_target in en_content and "Awareness Network Architecture" not in en_content:
    en_content = en_content.replace(en_target, en_injection)
    with open(en_file, "w", encoding="utf-8") as f:
        f.write(en_content)
    print("M1 en deck updated.")
else:
    print("Cannot find target in EN deck or already injected.")
