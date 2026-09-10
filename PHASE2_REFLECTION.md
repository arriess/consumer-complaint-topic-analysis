# Phase 2 - Development / Reflection

During Phase 2, I implemented and executed the NLP workflow on the selected CFPB sample. The file contained 5,000 complaints. After removing 683 exact duplicate narratives and one record that became empty during preprocessing, 4,316 documents remained. Cleaning included lowercasing, removal of URLs, e-mail artefacts, redaction placeholders, punctuation and numeric noise, regex tokenization, and English stop-word removal.

I compared Bag of Words and TF-IDF using the same 5,000-feature unigram/bigram vocabulary. Both matrices were 98.82% sparse, but Bag of Words stores counts whereas TF-IDF reweights common terms. LDA was applied to Bag of Words and LSA/TruncatedSVD to TF-IDF. Candidate topic counts 4-8 were evaluated using NPMI coherence, topic diversity and method-specific diagnostics. The selected solutions were 8 LDA topics (NPMI 0.2411) and 4 LSA components (NPMI 0.3421). LDA produced more granular themes; its largest topic, payment/loan/account-status reporting, represented 53.82% of documents. LSA produced broader latent dimensions.

An implementation problem was unavailable NLTK WordNet resources, so I replaced lemmatization with deterministic offline preprocessing. The sample contains only incorrect credit-reporting complaints and repeated legal/template language, limiting generalization.

GitHub: https://github.com/arriess/consumer-complaint-topic-analysis
