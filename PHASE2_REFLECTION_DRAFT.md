# Phase 2 Reflection Draft

During Phase 2, the selected CFPB complaint sample was validated and the text
analysis pipeline was implemented reproducibly. The file contained 5,000
complaints, but 683 exact duplicate narratives were removed and one additional
record became empty after preprocessing, leaving 4,316 documents for modeling.
A key discovery was that the sample is not a cross-section of all CFPB
complaints: every record concerns incorrect information on a credit report.
The project scope was therefore refined to recurring subthemes within this
specific complaint category.

Two vectorization techniques were compared: Bag of Words and TF-IDF. LDA was
applied to the Bag-of-Words matrix and LSA/TruncatedSVD to TF-IDF. Candidate
topic counts from four to eight were tested using NPMI coherence and topic
diversity, together with LDA perplexity and LSA explained variance. The executed
selection produced an eight-topic LDA solution and a four-component LSA
solution. LDA gave more granular themes and probabilistic topic prevalence,
while LSA produced broader latent dimensions.

One implementation issue was the planned WordNet lemmatization, because the
execution environment could not retrieve NLTK corpora. The final preprocessing
was changed to a deterministic offline procedure using scikit-learn stop words.
A further limitation is repeated legal/template language, which can influence
the discovered topics even after exact duplicates are removed.

GitHub repository: https://github.com/arriess/consumer-complaint-topic-analysis
