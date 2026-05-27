import sys

with open('mt_lnn_arxiv.tex', 'r', encoding='utf-8') as f:
    en_text = f.read()

needle_table_en = r'''
\begin{table}[ht]
\centering
\caption{Needle-in-a-Haystack retrieval accuracy on TinyLlama-1.1B. Base and MT-Adapter demonstrate flawless retrieval up to 4096 context with only minor latency degradation for the MT-Adapter.}
\label{tab:needle}
\begin{tabular}{lccccc}
\toprule
Variant & Context & Depth & Exact Match & Contains & Tok/s \\
\midrule
Base TinyLlama & 1024--2048 & All & 1.000 & 1.000 & $\sim$800 \\
MT-Adapter & 1024--2048 & All & \textbf{1.000} & \textbf{1.000} & $\sim$670 \\
\midrule
Base (RoPE scaled) & 4096 & All & 1.000 & 1.000 & $\sim$580 \\
MT-Adapter (RoPE) & 4096 & All & \textbf{1.000} & \textbf{1.000} & $\sim$545 \\
\bottomrule
\end{tabular}
\end{table}
'''

en_text = en_text.replace(
    'rather than architectural capability constraints.',
    'rather than architectural capability constraints (see Table~\\ref{tab:needle}).'
)

if needle_table_en not in en_text:
    idx = en_text.find('without disrupting their zero-shot instruction following or long-range reasoning capabilities.')
    idx = en_text.find('\n', idx)
    en_text = en_text[:idx+1] + needle_table_en + en_text[idx+1:]

with open('mt_lnn_arxiv.tex', 'w', encoding='utf-8') as f:
    f.write(en_text)

with open('mt_lnn_arxiv_zh.tex', 'r', encoding='utf-8') as f:
    zh_text = f.read()

wt103_zh = r'''\begin{table}[h]
\centering
\caption{WikiText-103 词级别困惑度与整合度评估比较。最好的结果已\textbf{加粗}。}
\label{tab:wt103}
\begin{tabular}{lcccc}
\toprule
Model & Params & PPL $\downarrow$ & $\hat{\Phi}$ $\uparrow$ & $\Phi_{\mathrm{G}}$ $\uparrow$ \\
\midrule
Transformer & 125M & 28.6 & 0.17 & 0.48 \\
LNN & 121M & 31.2 & 0.24 & 0.71 \\
MT-LNN-nGWT & 122M & 30.1 & 0.32 & 1.19 \\
\textbf{MT-LNN} & 128M & \textbf{24.4} & \textbf{0.38} & \textbf{1.74} \\
\bottomrule
\end{tabular}
\end{table}'''

lra_zh = r'''\begin{table}[h]
\centering
\caption{长距离竞技场 (LRA) 准确率 (\%). 最好的结果已\textbf{加粗}。}
\label{tab:lra}
\begin{tabular}{lccccc}
\toprule
Model & ListOps & Text & Retrieval & Image & Pathfinder \\
\midrule
Transformer & 36.3 & 64.2 & 57.4 & 42.8 & 71.7 \\
Linear Attn & 38.4 & 64.0 & 58.1 & 43.1 & 72.5 \\
S4 & 59.6 & 76.1 & \textbf{65.2} & 53.0 & 86.0 \\
\midrule
\textbf{MT-LNN} & \textbf{61.2} & \textbf{77.5} & 64.8 & \textbf{54.4} & \textbf{89.2} \\
\bottomrule
\end{tabular}
\end{table}'''

needle_zh = r'''\begin{table}[h]
\centering
\caption{TinyLlama-1.1B 大海捞针 (Needle-in-a-Haystack) 准确率及吞吐量。微管适配器在达到 4096 长度的 100\% 准确率的同时，仅有小幅延迟下降。}
\label{tab:needle}
\begin{tabular}{lccccc}
\toprule
模型/变体 & 上下文长度 & 测试深度 & 精确匹配 & 包含测试 & 吞吐量 (Tok/s) \\
\midrule
基础模型 (Base) & 1024--2048 & 全部 & 1.000 & 1.000 & $\sim$800 \\
MT微管适配器 & 1024--2048 & 全部 & \textbf{1.000} & \textbf{1.000} & $\sim$670 \\
\midrule
基线 (开启 RoPE) & 4096 & 全部 & 1.000 & 1.000 & $\sim$580 \\
MT微管适配器 (RoPE) & 4096 & 全部 & \textbf{1.000} & \textbf{1.000} & $\sim$545 \\
\bottomrule
\end{tabular}
\end{table}'''

anes_zh = r'''\begin{table}[h]
\centering
\caption{模拟麻醉测试下的 $\hat{\Phi}$ 与 $\Phi_{\mathrm{G}}$ 坍缩结果 ($\kappa=1\to10$ 浓度)。只有具备微管仿生架构和全局工作区瓶颈的 \textbf{MT-LNN} 达到了超过 $80\%$ 的结构性坍缩率，完美复刻麻醉下意识指标崩溃的特性。}
\label{tab:anesth}
\resizebox{\textwidth}{!}{
\begin{tabular}{lccccccc}
\toprule
 & \multicolumn{3}{c}{$\hat{\Phi}$ (kNN估计)} & \multicolumn{3}{c}{$\Phi_{\mathrm{G}}$ (高斯估计)} & \\
\cmidrule{2-4}\cmidrule{5-7}
架构模型 & 正常 ($\kappa{=}1$) & 麻醉 ($\kappa{=}10$) & 坍缩率 & 正常 & 麻醉 & 坍缩率 & 结论 \\
\midrule
Transformer & 0.17 & 0.16 & 5.9\% & 0.48 & 0.45 & 6.3\% & \textcolor{red}{$\times$} \\
LNN & 0.24 & 0.22 & 8.3\% & 0.71 & 0.64 & 9.9\% & \textcolor{red}{$\times$} \\
MT-LNN-nGWT & 0.32 & 0.11 & 65.6\% & 1.19 & 0.41 & 65.5\% & \textcolor{red}{$\times$} \\
\textbf{MT-LNN} & \textbf{0.38} & \textbf{0.04} & \textbf{89.5\%} & \textbf{1.74} & \textbf{0.21} & \textbf{87.9\%} & \textbf{通过} \\
\bottomrule
\end{tabular}}
\end{table}'''

zh_text = zh_text.replace(
    '这种提升同时也延伸到长距离竞技场 (Long-Range Arena) 的 Pathfinder 和 Selective Copy 长文本任务中。',
    '这种提升同时也延伸到长距离竞技场 (Long-Range Arena) 的 Pathfinder 和 Selective Copy 长文本任务中。\n\n' + wt103_zh + '\n\n' + lra_zh + '\n'
)

zh_text = zh_text.replace(
    '未破坏基础模型原有的指令跟随与推理能力。',
    '未破坏基础模型原有的指令跟随与推理能力 (见表~\\ref{tab:needle})。'
)

idx_needle = zh_text.find('完美兼容现代大语言模型原生及外推扩展后的长下文特征状态。')
idx_needle = zh_text.find('\n', idx_needle)
zh_text = zh_text[:idx_needle+1] + '\n' + needle_zh + '\n' + zh_text[idx_needle+1:]

idx_anes = zh_text.find('呈现约 5\\% 左右的毫无规律的数值抖动。这通过计算的层面为假说提供了极为生动的验证。')
idx_anes = zh_text.find('\n', idx_anes)
zh_text = zh_text[:idx_anes+1] + '\n' + anes_zh + '\n' + zh_text[idx_anes+1:]

with open('mt_lnn_arxiv_zh.tex', 'w', encoding='utf-8') as f:
    f.write(zh_text)

print('Fully updated')
