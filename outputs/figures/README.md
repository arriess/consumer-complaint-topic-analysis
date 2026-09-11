# Generated figures

This directory is populated when the analysis pipeline is executed.

Generated files:

- `topic_count_npmi_coherence.png` - NPMI coherence across candidate topic/component counts.
- `lda_topic_prevalence.png` - LDA document-assignment shares by topic.
- `lsa_topic_prevalence.png` - descriptive LSA document-assignment shares by component using the largest absolute component loading.

The PNG files are generated artifacts and are excluded by `.gitignore`; running `python run_analysis.py` recreates them from the selected public dataset and the documented model settings.
