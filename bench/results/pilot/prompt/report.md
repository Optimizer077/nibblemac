| variant | tokens (18 req.) | saved | with schema cache | max \|dp\| ref | same top ref | mean \|dp\| ref | max \|dp\| base | same top base | identical? |
|---|---|---|---|---|---|---|---|---|---|
| base | 3703 | 0.0% | 0.0% | 0.0094 | 51/52 | 0.0023 | 0.0000 | 52/52 | yes |
| multi | 4011 | -8.3% | -8.3% | 0.1494 | 51/52 | 0.0280 | 0.1399 | 50/52 | no |
| multi_ans | 4079 | -10.2% | -10.2% | 0.1534 | 50/52 | 0.0294 | 0.1439 | 49/52 | no |
| schema | 4011 | -8.3% | 52.3% | 0.5934 | 47/52 | 0.0944 | 0.5937 | 46/52 | no |
| schema_q | 4761 | -28.6% | 32.1% | – | – | – | – | – | not run |
| compact_a | 3547 | 4.2% | 4.2% | 0.1453 | 51/52 | 0.0159 | 0.1498 | 52/52 | no |
| compact_b | 3334 | 10.0% | 10.0% | 0.1735 | 50/52 | 0.0338 | 0.1762 | 51/52 | no |
| compact_c | 3282 | 11.4% | 11.4% | 0.2617 | 48/52 | 0.0606 | 0.2629 | 49/52 | no |
| compact_d | 3703 | 0.0% | 0.0% | – | – | – | – | – | not run |
| shared | 3657 | 1.2% | 1.2% | – | – | – | – | – | not run |

- base: largest |dp| vs ref: contract_clause/favours 0.0094, math_word/answer 0.0080, agent_guardrail_email/risk 0.0070
- multi: largest |dp| vs ref: contract_clause/favours 0.1494, agent_guardrail/risk 0.1347, translation_quality/quality 0.0982
- multi_ans: largest |dp| vs ref: contract_clause/favours 0.1534, agent_guardrail/risk 0.1132, sentiment_review/stars 0.1058
- schema: largest |dp| vs ref: long_incident/detection 0.5934, country_wide/country 0.3864, code_review/bug 0.2627
- compact_a: largest |dp| vs ref: router_wide/category 0.1453, contract_clause/favours 0.0661, math_word/answer 0.0582
- compact_b: largest |dp| vs ref: router_wide/category 0.1735, math_word/needs_arithmetic 0.1454, code_review/bug 0.1392
- compact_c: largest |dp| vs ref: code_review/bug 0.2617, long_incident/process_gap 0.2150, math_word/needs_arithmetic 0.1813
