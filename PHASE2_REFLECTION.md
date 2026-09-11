# Phase 2 - Development / Reflection

During Phase 2, I implemented and executed the planned NLP pipeline on the selected CFPB complaint sample. The source contained 5,000 complaints. After removing 683 exact duplicate narratives and one record that became empty after preprocessing, 4,316 documents remained. Cleaning included lowercasing, removal of URLs, e-mail artefacts, redaction placeholders, punctuation and numeric noise, regex tokenization, and English stop-word removal.

I compared Bag of Words and TF-IDF with the same 5,000-feature unigram/bigram vocabulary. Both matrices were 98.82% sparse, while their values differed: raw term counts versus TF-IDF weights. LDA was fitted to Bag of Words and LSA/TruncatedSVD to TF-IDF. Candidate topic counts from 4 to 8 were compared using NPMI coherence, topic diversity, LDA perplexity and LSA explained variance. The selected solutions were 8 LDA topics (NPMI 0.2411) and 4 LSA components (NPMI 0.3421). LDA produced more granular themes; its largest topic represented 53.82% of documents.

A key development issue was that validation showed the sample was narrower than a general complaint corpus: every record concerned incorrect information on a credit report. I therefore restricted interpretation to subthemes within this category. Repeated legal/template language may still influence topic structure even after exact duplicates are removed.

GitHub: https://github.com/arriess/consumer-complaint-topic-analysis
