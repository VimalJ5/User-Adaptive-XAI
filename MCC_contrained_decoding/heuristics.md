# Readability Heuristics — Reference Table

| Heuristic | What it measures | Source formula | Status |
|---|---|---|---|
| **Syllable count** (avg syllables/word) | Phonological difficulty; high in biomedical text due to Greek/Latin roots | Flesch Reading Ease (1948); Flesch-Kincaid Grade Level (1975) | Existing |
| **Polysyllabic word flag** (words ≥ 3 syllables) | Isolates hard words by threshold — dilution by short words makes the average a weaker signal | SMOG Index (McLaughlin, 1969); Gunning Fog (Gunning, 1952) | New |
| **Dale-Chall unfamiliar words** (not in ~3k familiar-word list) | Replaces the length≥5 rare-word proxy; catches short hard words ("renal", "sepsis") that length heuristics miss | Dale-Chall Readability Formula (Dale & Chall, 1948; revised Chall & Dale, 1995) | Replaces "rare word" |
| **Characters per word** (avg chars/word) | Orthogonal to syllables; captures long abbreviations and compound clinical terms | Automated Readability Index (Senter & Smith, 1967); Coleman-Liau Index (1975) | New |
| **Clause markers** (subordinating conjunctions) | Multi-clause sentences increase working memory load; each connective signals a new embedded clause | Coh-Metrix connective density (McNamara et al., 2014) | Existing |
| **Avg sentence length** (words/sentence) | Most consistent predictor across all major formulas; longer sentences increase parse depth | Flesch-Kincaid (1975); Gunning Fog (1952); Dale-Chall (1995); ARI (1967) | New |
| **Generation length** (total word count) | Length budget — increases EOS preference as output grows; not a readability signal per se | Custom; motivated by Grice's Maxim of Quantity | Existing |

## References

- Dale, E., & Chall, J. S. (1948). A formula for predicting readability. *Educational Research Bulletin*, 27(1), 11–28. Revised as Chall & Dale (1995). *Readability Revisited*. Brookline Books.
- Coleman, M., & Liau, T. L. (1975). A computer readability formula. *Journal of Applied Psychology*, 60(2), 283–284.
- Flesch, R. (1948). A new readability yardstick. *Journal of Applied Psychology*, 32(3), 221–233.
- Gunning, R. (1952). *The Technique of Clear Writing*. McGraw-Hill.
- Kincaid, J. P., et al. (1975). *Derivation of new readability formulas for Navy enlisted personnel*. Naval Technical Training Command.
- McLaughlin, G. H. (1969). SMOG grading — a new readability formula. *Journal of Reading*, 12(8), 639–646.
- McNamara, D. S., et al. (2014). *Automated Evaluation of Text and Discourse with Coh-Metrix*. Cambridge University Press.
- Senter, R. J., & Smith, E. A. (1967). *Automated readability index*. Wright-Patterson Air Force Base.