# GenDLN

This is the repository of the paper [**GenDLN: Evolutionary Algorithm-Based Stacked LLM Framework for Joint Prompt Optimization**](https://openreview.net/pdf?id=64xhaOC8gE) by Pia Chouayfati, Niklas Herbster, Ábel Domonkos Sáfrán, and Matthias Grabmair.

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
    ├── framework/



Note: We provie an "LLM-Safe" MRPC dataset. Details can be found in [Appendix P](https://openreview.net/pdf?id=64xhaOC8gE) of the paper.

#### Citation

    @inproceedings{
    chouayfati2025gendln,
    title={Gen{DLN}: Evolutionary Algorithm-Based Stacked {LLM} Framework for Joint Prompt Optimization},
    author={Pia Chouayfati and Niklas Herbster and {\'A}bel Domonkos S{\'a}fr{\'a}n and Matthias Grabmair},
    booktitle={Association of Computational Linguistics 2025 SRW},
    year={2025},
    url={https://openreview.net/forum?id=64xhaOC8gE}
    }

