## MultilingualList
MultilingualList found the wordfreq[^1] top 3,702 English words and their synsets[^2][^3][^4][^5]. From those synsets, that found words in English, French and Spanish. The French Noun Gender Dataset[^6] and AnCora GitHub Repository[^7] provided genders of French and Spanish nouns when possible.

Some synsets were omitted due to

* a stopword match[^8]
* the dataset having the wrong word for a synset ID;
* a part of speech other than adjective, verb, adverb and noun;
* duplicates;
* no French or Spanish words;
* proper nouns;
* numerical results;
* profanity[^9];
* and obscurity or archaism.

All that had 2,000 synset IDs remaining with their part of speech and English, French and Spanish words turned into flashcards.

```python
'''
./citations - - - - OMW citation output directory
    /
./csv - - - - - - - synset output directory
    /synsets.csv
./png - - - - - - - flashcard output directory
    /[synset ID].png
./resources
    /external
        /es_ancora-ud-dev.conllu.txt - - AnCora GitHub Repository
        /french-noun-gender.csv  - - - - French Noun Gender Dataset
    /internal
        /gender-dependent.csv - - clarification
            fr,tour,revolution,masculine
            fr,tour,tower,feminine
            fr,aide,assistant,epicene
            fr,aide,care,feminine
        /omit.csv - - - - - - - - synsets and words to omit
            09328904-n,fr,laque <- Skipped "laque" since that's
                                   "lacquer" or "hair spray" and
                                   not "lake". "fr" and "laque" are
                                   optional if the entire synset ID
                                   were to get skipped.
'''

ML = MultilingualList()
ML.synset_csv()
ML.flashcards()
ML.citations()
```

## BibTeX
<details>
<summary>OMW v2 English</summary>

```bibtex
@book{_Fellbaum:1998,
  booktitle =	 "{WordNet}: An Electronic Lexical Database",
  address =	 "Cambridge, MA",
  editor =	 "Fellbaum, Christiane",
  publisher =	 "MIT Press",
  year =	 1998,
}
```

</details>

<details>
<summary>OMW v2 French</summary>

```bibtex
@InProceedings{Sagot:Fiser:2008,
  author =	 {Benoît Sagot and Darja Fišer},
  title =	 {Building a free {French} wordnet from multilingual
                  resources},
  booktitle =	 {Proceedings of the Sixth International Language
                  Resources and Evaluation (LREC'08)},
  year =	 2008,
  editor =	 {European Language Resources Association (ELRA)},
  address =	 {Marrakech, Morocco}
}
```

</details>

<details>
<summary>OMW v2 Spanish</summary>

```bibtex
@InProceedings{Gonzalez-Agirre:Laparra:Rigau:2012,
  author =	 "Aitor Gonzalez-Agirre and Egoitz Laparra and German
                  Rigau",
  title =	 "Multilingual Central Repository version 3.0:
                  upgrading a very large lexical knowledge base",
  booktitle =	 "Proceedings of the 6th Global WordNet Conference
                  (GWC 2012)",
  year =	 2012,
  address =	 "Matsue",
}
```

</details>

[^1]: [wordfreq 3.1.1](https://github.com/rspeer/wordfreq) - Apache License 2.0

[^2]: [Open Multilingual Wordnet (OMW) 2.0](https://omwn.org/omw2.html) - Apache License 2.0

[^3]: OMW v2 English - Princeton WordNet 3.0

[^4]: OMW v2 French - CeCILL-C

[^5]: OMW v2 Spanish - CC BY 3.0

[^6]: [French Noun Gender Dataset](https://m-or-f.com/french-noun-gender-dataset) - Lesser General Public License For Linguistic Resources (LGPL-LR)

[^7]: [AnCora GitHub Repository v2.18](https://github.com/UniversalDependencies/UD_Spanish-AnCora) - Creative Commons License Attribution 4.0 International

[^8]: [Natural Language Toolkit (NLTK) 3.10.3](https://www.nltk.org/) - Apache License 2.0

[^9]: [better-profanity 0.7.0](https://pypi.org/project/better-profanity/) - MIT License
