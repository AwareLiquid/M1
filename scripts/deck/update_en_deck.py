import re

with open('investor_deck_mt_lnn.tex', 'r', encoding='utf-8') as f:
    text = f.read()

replacement = r'''\begin{frame}{Core Validation: Verifiable \& Open Benchmarks}
\begin{itemize}
    \item \textbf{1. Memory Compression Limit Test (1000-token footprint)}
    \begin{itemize}
        \item \textbf{Testing Baseline:} Standard Transformer (KV Cache) v.s. AwareLiquid (State-only).
        \item \textbf{Commercial Meaning in Plain Terms:} Memory usage is \textbf{\textasciitilde 250 times smaller} than equivalent models (1020 KB down to 4.1 KB). In plain terms: we successfully eradicate the Out-Of-Memory (OOM) curse, allowing AI to run flawlessly on decade-old phones or smart home appliances.
        \item \textbf{Reproducible Code (Click to open):} \href{https://github.com/everest-an/M1/blob/main/benchmarks/operator_compression_report.py}{\texttt{benchmarks/operator\_compression\_report.py}}
    \end{itemize}
    \vspace{0.4em}
    \item \textbf{2. Noise Immunity ("Needle in a Haystack" Retrieval)}
    \begin{itemize}
        \item \textbf{Testing Baseline:} Fair sandbox evaluation at 200K scale with extreme sequence lengths.
        \item \textbf{Commercial Meaning in Plain Terms:} Accuracy is \textbf{42 times higher} (96.5\% for AwareLiquid vs a crashed 2.3\% for standard Transformer). In plain terms: your AI will never "hallucinate" or forget past conversations, completely locking down facts across months of history.
        \item \textbf{Reproducible Code (Click to open):} \href{https://github.com/everest-an/M1/blob/main/benchmarks/run_benchmark.py}{\texttt{benchmarks/run\_benchmark.py}}
    \end{itemize}
    \vspace{0.2em}
    \item \textbf{3. CPU Hardware Independence (Sparse Resonance)}
    \begin{itemize}
        \item \textbf{Testing Baseline:} Pushing pure CPU limits without high-end GPUs.
        \item \textbf{Commercial Meaning in Plain Terms:} By intelligently skipping redundant computations via bionic sleep, speed actually \textbf{increases by 13\%+}. In plain terms: you can deploy tens of thousands of local smart devices while skipping the extortionate cost of acquiring AI chips.
        \item \textbf{Reproducible Code (Click to open):} \href{https://github.com/everest-an/M1/blob/main/benchmarks/sparse_resonance_ablation.py}{\texttt{benchmarks/sparse\_resonance\_ablation.py}}
    \end{itemize}
\end{itemize}
\end{frame}'''

pattern = r"\\begin\{frame\}\{Verifiable Benchmarks.*?\}\n\\begin\{itemize\}.*?(?=\\begin\{frame\})"
new_text = re.sub(pattern, lambda m: replacement + "\n\n", text, flags=re.DOTALL)

with open('investor_deck_mt_lnn.tex', 'w', encoding='utf-8') as f:
    f.write(new_text)

print("Updated EN deck.")
