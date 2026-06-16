# Keyword Analysis – Syllabi 2020–2026

## Corpus overview

| Metric                               | Value       |
| ------------------------------------ | ----------- |
| Total syllabi                        | 347         |
| Years covered                        | 2020 – 2026 |
| Total tokens (after stopword filter) | 384,241     |
| Unique tokens                        | 27,576      |

## Top 100 words (corpus-wide)

| Rank | Word           | Total count |
| ---- | -------------- | ----------- |
| 1    | design         | 6,815       |
| 2    | architecture   | 6,451       |
| 3    | climate        | 3,851       |
| 4    | change         | 2,206       |
| 5    | environmental  | 2,136       |
| 6    | development    | 1,994       |
| 7    | urban          | 1,944       |
| 8    | architectural  | 1,916       |
| 9    | building       | 1,806       |
| 10   | work           | 1,742       |
| 11   | community      | 1,316       |
| 12   | social         | 1,303       |
| 13   | project        | 1,236       |
| 14   | systems        | 1,203       |
| 15   | energy         | 1,194       |
| 16   | material       | 1,166       |
| 17   | environment    | 1,146       |
| 18   | city           | 1,059       |
| 19   | sustainable    | 932         |
| 20   | site           | 873         |
| 21   | public         | 872         |
| 22   | center         | 865         |
| 23   | future         | 864         |
| 24   | practice       | 860         |
| 25   | projects       | 837         |
| 26   | built          | 835         |
| 27   | landscape      | 765         |
| 28   | global         | 761         |
| 29   | planning       | 760         |
| 30   | sustainability | 757         |
| 31   | materials      | 742         |
| 32   | infrastructure | 735         |
| 33   | construction   | 716         |
| 34   | society        | 702         |
| 35   | cultural       | 692         |
| 36   | develop        | 679         |
| 37   | world          | 672         |
| 38   | carbon         | 669         |
| 39   | strategies     | 658         |
| 40   | architects     | 656         |
| 41   | critical       | 655         |
| 42   | human          | 632         |
| 43   | justice        | 628         |
| 44   | history        | 622         |
| 45   | ecological     | 622         |
| 46   | including      | 615         |
| 47   | analysis       | 614         |
| 48   | water          | 606         |
| 49   | study          | 606         |
| 50   | communities    | 600         |
| 51   | state          | 599         |
| 52   | american       | 596         |
| 53   | cities         | 589         |
| 54   | learning       | 589         |
| 55   | buildings      | 576         |
| 56   | technology     | 575         |
| 57   | issues         | 571         |
| 58   | time           | 566         |
| 59   | local          | 558         |
| 60   | impact         | 556         |
| 61   | experience     | 548         |
| 62   | challenges     | 546         |
| 63   | assistant      | 546         |
| 64   | scale          | 542         |
| 65   | space          | 538         |
| 66   | housing        | 537         |
| 67   | understanding  | 525         |
| 68   | science        | 523         |
| 69   | architect      | 516         |
| 70   | knowledge      | 511         |
| 71   | associate      | 507         |
| 72   | year           | 503         |
| 73   | professional   | 503         |
| 74   | performance    | 495         |
| 75   | context        | 488         |
| 76   | spatial        | 480         |
| 77   | resources      | 461         |
| 78   | data           | 456         |
| 79   | approach       | 455         |
| 80   | health         | 455         |
| 81   | resilience     | 452         |
| 82   | master         | 445         |
| 83   | years          | 445         |
| 84   | methods        | 443         |
| 85   | practices      | 443         |
| 86   | conditions     | 442         |
| 87   | economic       | 441         |
| 88   | life           | 440         |
| 89   | natural        | 440         |
| 90   | plan           | 431         |
| 91   | process        | 431         |
| 92   | include        | 424         |
| 93   | module         | 424         |
| 94   | texas          | 417         |
| 95   | address        | 414         |
| 96   | art            | 411         |
| 97   | level          | 403         |
| 98   | national       | 403         |
| 99   | nature         | 399         |
| 100  | current        | 398         |

## Biggest movers

_(comparing 2020 → 2026, frequency per 1 000 words)_

### Rising words

| Word          | Δ freq/1 000 |
| ------------- | ------------ |
| architectural | +4.27        |
| design        | +4.10        |
| material      | +2.96        |
| environmental | +2.43        |
| community     | +2.21        |
| systems       | +2.08        |
| performance   | +1.90        |
| building      | +1.47        |
| site          | +1.37        |
| analysis      | +1.30        |

### Declining words

| Word         | Δ freq/1 000 |
| ------------ | ------------ |
| change       | -5.34        |
| architecture | -3.91        |
| center       | -2.03        |
| issues       | -1.89        |
| environment  | -1.62        |
| history      | -1.52        |
| american     | -1.25        |
| city         | -1.23        |
| global       | -1.17        |
| nature       | -1.14        |

## Notes

- Frequencies are normalised per 1 000 words to account for varying syllabus lengths.
- `stopwords.txt` (general English stopwords) was applied during PDF preprocessing.
- `stopwords_2.txt` (academic/institutional boilerplate) was applied here so results reflect climate/design language rather than university context.
- All output files are in `1_keyword_analysis_output/`.
- Log-likelihood (G²) tests whether each word’s frequency in a given year departs
  significantly from its expected rate; see `log_likelihood_plain_english.md` for interpretation.
