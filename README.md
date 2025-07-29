# GenDLN

This is the repository of the paper [**GenDLN: Evolutionary Algorithm-Based Stacked LLM Framework for Joint Prompt Optimization**](https://aclanthology.org/2025.acl-srw.92/) by Pia Chouayfati, Niklas Herbster, Ábel Domonkos Sáfrán, and Matthias Grabmair.

### Organization

    GenDLN/
    ├── datasets/
    │   ├── claudette
    │   ├── llm_safe_mrpc
    ├── ga_post_preocessing_R/
    │   ├── GA_Runs.Rmd      # R Markdown file for generating reports
    │   ├── GA_Runs.html     # Rendered example on dummy_logs
    │   ├── R_helpers      # Normalization, parsing, processing, plotting... scripts
    │   ├── dummy_logs     # example logs to run the GA_Runs.Rmd
    ├── genetic_dln/       # GenDLN framework



Note: We provie an "LLM-Safe" MRPC dataset. Details can be found in [Appendix P](https://aclanthology.org/2025.acl-srw.92.pdf) of the paper.

#### Citation (BibTex)

    @inproceedings{chouayfati-etal-2025-gendln,
    title = "{G}en{DLN}: Evolutionary Algorithm-Based Stacked {LLM} Framework for Joint Prompt Optimization",
    author = "Chouayfati, Pia  and
      Herbster, Niklas  and
      S{\'a}fr{\'a}n, {\'A}bel Domonkos  and
      Grabmair, Matthias",
    editor = "Zhao, Jin  and
      Wang, Mingyang  and
      Liu, Zhu",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 4: Student Research Workshop)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-srw.92/",
    pages = "1171--1212",
    ISBN = "979-8-89176-254-1",
    abstract = "With Large Language Model (LLM)-based applications becoming more common due to strong performance across many tasks, prompt optimization has emerged as a way to extract better solutions from frozen, often commercial LLMs that are not specifically adapted to a task. LLM-assisted prompt optimization methods provide a promising alternative to manual/human prompt engineering, where LLM ``reasoning'' can be used to make them optimizing agents. However, the cost of using LLMs for prompt optimization via commercial APIs remains high, especially for heuristic methods like evolutionary algorithms (EAs), which need many iterations to converge, and thus, tokens, API calls, and rate-limited network overhead. We propose GenDLN, an open-source, efficient genetic algorithm-based prompt pair optimization framework that leverages commercial API free tiers. Our approach allows teams with limited resources (NGOs, non-profits, academics, ...) to efficiently use commercial LLMs for EA-based prompt optimization. We conduct experiments on CLAUDETTE for legal terms of service classification and MRPC for paraphrase detection, performing in line with selected prompt optimization baselines, at no cost."}

