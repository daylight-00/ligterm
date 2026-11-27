# Historical test: the measured performance of the evaluated refits

The two studies are a previously reserved, once-evaluated historical test: they were held out of model development and evaluated once, after the models and the evaluation rules were fixed. The result is the measured performance of the evaluated refits (group-disjoint and lab-excluded regimes, with the 50%-identity grouping as the sensitivity) beside the development results. It is not independent evidence for the final predictor, there is no pass rule, and the paired differences are descriptions. The pooled value of a cell is the unweighted mean of the two study-level values; the interval is the 95% percentile bootstrap over sequence groups across both studies. Both studies share peptides with the development studies and with each other, and their units are ordinary training units of every development run.

## Regime group_disjoint

| study | training units | proteins | training positives N / C | evaluation proteins (bond level N / C) | candidate positives |
|---|---|---|---|---|---|
| 42155283 | 44764 | 5589 | 29685 / 29870 | 4600 / 4455 | 33100 |
| 35051231 | 53316 | 3964 | 29387 / 29693 | 7668 / 7666 | 24163 |

Pooled values with 95% intervals (study values below):

| model | Bond-AUROC N | Bond-AUROC C | Local-AUC N | Local-AUC C | Peptide-AUROC (both_bonds) | Length-matched AUC (both_bonds) | Near-miss AUC (both_bonds) | Peptide-AUROC (both_bonds_and_length) | Length-matched AUC (both_bonds_and_length) | Near-miss AUC (both_bonds_and_length) |
|---|---|---|---|---|---|---|---|---|---|---|
| LigTerm-GB | 0.6578 [0.6538, 0.6632] | 0.6400 [0.6359, 0.6452] | 0.6494 [0.6455, 0.6555] | 0.6208 [0.6171, 0.6259] | 0.6732 [0.6683, 0.6787] | 0.6688 [0.6640, 0.6741] | 0.6766 [0.6727, 0.6827] | 0.7745 [0.7702, 0.7784] | 0.6688 [0.6640, 0.6741] | 0.7131 [0.7092, 0.7194] |
| LR-Both8 | 0.6459 [0.6420, 0.6515] | 0.6064 [0.6024, 0.6109] | 0.6476 [0.6441, 0.6534] | 0.5989 [0.5945, 0.6051] | 0.6649 [0.6606, 0.6702] | 0.6629 [0.6585, 0.6684] | 0.6690 [0.6644, 0.6762] | 0.7783 [0.7744, 0.7823] | 0.6629 [0.6585, 0.6684] | 0.7080 [0.7034, 0.7156] |
| LR-Both16 | 0.6508 [0.6469, 0.6560] | 0.6332 [0.6290, 0.6382] | 0.6491 [0.6452, 0.6553] | 0.6214 [0.6175, 0.6267] | 0.6677 [0.6629, 0.6732] | 0.6612 [0.6566, 0.6667] | 0.6777 [0.6734, 0.6843] | 0.7667 [0.7624, 0.7710] | 0.6612 [0.6566, 0.6667] | 0.7112 [0.7068, 0.7185] |
| LR-Both16-CV | 0.6510 [0.6472, 0.6563] | 0.6321 [0.6279, 0.6371] | 0.6489 [0.6449, 0.6552] | 0.6215 [0.6175, 0.6269] | 0.6677 [0.6629, 0.6733] | 0.6611 [0.6564, 0.6666] | 0.6776 [0.6732, 0.6843] | 0.7652 [0.7609, 0.7696] | 0.6611 [0.6564, 0.6666] | 0.7105 [0.7062, 0.7179] |
| Paul-style motif | 0.6128 [0.6087, 0.6172] | 0.5668 [0.5631, 0.5712] | 0.6348 [0.6314, 0.6397] | 0.5901 [0.5865, 0.5954] | 0.6358 [0.6315, 0.6403] | 0.6353 [0.6310, 0.6397] | 0.6574 [0.6536, 0.6632] | 0.7800 [0.7760, 0.7830] | 0.6353 [0.6310, 0.6397] | 0.6964 [0.6926, 0.7025] |

Study 42155283:

| model | Bond-AUROC N | Bond-AUROC C | Local-AUC N | Local-AUC C | Peptide-AUROC (both_bonds) | Length-matched AUC (both_bonds) | Near-miss AUC (both_bonds) | Peptide-AUROC (both_bonds_and_length) | Length-matched AUC (both_bonds_and_length) | Near-miss AUC (both_bonds_and_length) |
|---|---|---|---|---|---|---|---|---|---|---|
| LigTerm-GB | 0.7711 | 0.7423 | 0.7374 | 0.6904 | 0.8077 | 0.8006 | 0.7820 | 0.8530 | 0.8006 | 0.7904 |
| LR-Both8 | 0.7379 | 0.6790 | 0.7311 | 0.6515 | 0.7866 | 0.7845 | 0.7665 | 0.8436 | 0.7845 | 0.7850 |
| LR-Both16 | 0.7605 | 0.7291 | 0.7340 | 0.6899 | 0.7964 | 0.7878 | 0.7801 | 0.8486 | 0.7878 | 0.7896 |
| LR-Both16-CV | 0.7612 | 0.7272 | 0.7336 | 0.6902 | 0.7963 | 0.7879 | 0.7800 | 0.8485 | 0.7879 | 0.7893 |
| Paul-style motif | 0.6710 | 0.6159 | 0.7106 | 0.6392 | 0.7182 | 0.7173 | 0.7471 | 0.7856 | 0.7173 | 0.7396 |

Study 35051231:

| model | Bond-AUROC N | Bond-AUROC C | Local-AUC N | Local-AUC C | Peptide-AUROC (both_bonds) | Length-matched AUC (both_bonds) | Near-miss AUC (both_bonds) | Peptide-AUROC (both_bonds_and_length) | Length-matched AUC (both_bonds_and_length) | Near-miss AUC (both_bonds_and_length) |
|---|---|---|---|---|---|---|---|---|---|---|
| LigTerm-GB | 0.5445 | 0.5377 | 0.5614 | 0.5512 | 0.5387 | 0.5371 | 0.5712 | 0.6961 | 0.5371 | 0.6357 |
| LR-Both8 | 0.5539 | 0.5337 | 0.5641 | 0.5462 | 0.5431 | 0.5414 | 0.5715 | 0.7131 | 0.5414 | 0.6311 |
| LR-Both16 | 0.5411 | 0.5373 | 0.5641 | 0.5528 | 0.5391 | 0.5346 | 0.5753 | 0.6849 | 0.5346 | 0.6327 |
| LR-Both16-CV | 0.5409 | 0.5370 | 0.5641 | 0.5528 | 0.5391 | 0.5342 | 0.5751 | 0.6818 | 0.5342 | 0.6318 |
| Paul-style motif | 0.5546 | 0.5178 | 0.5589 | 0.5409 | 0.5534 | 0.5532 | 0.5677 | 0.7743 | 0.5532 | 0.6532 |

Descriptive paired differences (first minus second; pooled, with the interval; the per-study values follow):

| contrast | Bond-AUROC N | Bond-AUROC C | Local-AUC N | Local-AUC C | Length-matched AUC (both_bonds) | Near-miss AUC (both_bonds) | Peptide-AUROC (both_bonds) |
|---|---|---|---|---|---|---|---|
| LigTerm-GB minus LR-Both16 | 0.0070 [0.0056, 0.0087] | 0.0068 [0.0052, 0.0083] | 0.0004 [-0.0018, 0.0026] | -0.0006 [-0.0029, 0.0016] | 0.0076 [0.0060, 0.0090] | -0.0010 [-0.0032, 0.0005] | 0.0055 [0.0040, 0.0069] |
| &nbsp;&nbsp;42155283 | 0.0106 | 0.0132 | 0.0034 | 0.0005 | 0.0127 | 0.0020 | 0.0113 |
| &nbsp;&nbsp;35051231 | 0.0034 | 0.0004 | -0.0027 | -0.0016 | 0.0025 | -0.0041 | -0.0004 |
| LigTerm-GB minus LR-Both16-CV | 0.0068 [0.0053, 0.0084] | 0.0079 [0.0063, 0.0094] | 0.0005 [-0.0016, 0.0027] | -0.0007 [-0.0032, 0.0015] | 0.0077 [0.0061, 0.0091] | -0.0009 [-0.0032, 0.0006] | 0.0055 [0.0040, 0.0068] |
| &nbsp;&nbsp;42155283 | 0.0099 | 0.0152 | 0.0038 | 0.0002 | 0.0126 | 0.0020 | 0.0114 |
| &nbsp;&nbsp;35051231 | 0.0037 | 0.0006 | -0.0027 | -0.0016 | 0.0028 | -0.0039 | -0.0004 |
| LR-Both16-CV minus LR-Both16 | 0.0002 [0.0001, 0.0004] | -0.0011 [-0.0012, -0.0009] | -0.0002 [-0.0005, 0.0003] | 0.0001 [-0.0002, 0.0005] | -0.0001 [-0.0002, 0.0000] | -0.0001 [-0.0003, 0.0001] | 0.0000 [-0.0001, 0.0002] |
| &nbsp;&nbsp;42155283 | 0.0007 | -0.0019 | -0.0004 | 0.0002 | 0.0001 | -0.0000 | -0.0000 |
| &nbsp;&nbsp;35051231 | -0.0002 | -0.0003 | 0.0000 | -0.0000 | -0.0004 | -0.0002 | 0.0000 |
| LR-Both16 minus LR-Both8 | 0.0049 [0.0025, 0.0067] | 0.0268 [0.0240, 0.0297] | 0.0014 [-0.0005, 0.0034] | 0.0225 [0.0187, 0.0251] | -0.0017 [-0.0038, -0.0000] | 0.0086 [0.0066, 0.0102] | 0.0028 [0.0006, 0.0046] |
| &nbsp;&nbsp;42155283 | 0.0225 | 0.0501 | 0.0029 | 0.0384 | 0.0033 | 0.0135 | 0.0097 |
| &nbsp;&nbsp;35051231 | -0.0128 | 0.0036 | 0.0000 | 0.0066 | -0.0068 | 0.0037 | -0.0041 |
| LR-Both16 minus Paul-style motif | 0.0380 [0.0347, 0.0418] | 0.0664 [0.0626, 0.0705] | 0.0143 [0.0119, 0.0177] | 0.0313 [0.0282, 0.0342] | 0.0259 [0.0224, 0.0300] | 0.0203 [0.0182, 0.0230] | 0.0319 [0.0283, 0.0359] |
| &nbsp;&nbsp;42155283 | 0.0895 | 0.1132 | 0.0234 | 0.0507 | 0.0705 | 0.0330 | 0.0782 |
| &nbsp;&nbsp;35051231 | -0.0135 | 0.0195 | 0.0052 | 0.0118 | -0.0186 | 0.0076 | -0.0144 |

Selection (LR-Both16-CV: penalty by the rule on the training folds; LigTerm-GB: boosting iterations by the rule on the training folds): 42155283: N C=1.0, 100 it; C C=0.03, 100 it; 35051231: N C=3.0, 200 it; C C=0.3, 100 it

## Regime lab_excluded

| study | training units | proteins | training positives N / C | evaluation proteins (bond level N / C) | candidate positives |
|---|---|---|---|---|---|
| 42155283 | 14105 | 4148 | 11429 / 11535 | 4600 / 4455 | 33100 |
| 35051231 | 24787 | 3208 | 15297 / 15256 | 7668 / 7666 | 24163 |

Pooled values with 95% intervals (study values below):

| model | Bond-AUROC N | Bond-AUROC C | Local-AUC N | Local-AUC C | Peptide-AUROC (both_bonds) | Length-matched AUC (both_bonds) | Near-miss AUC (both_bonds) | Peptide-AUROC (both_bonds_and_length) | Length-matched AUC (both_bonds_and_length) | Near-miss AUC (both_bonds_and_length) |
|---|---|---|---|---|---|---|---|---|---|---|
| LigTerm-GB | 0.6162 [0.6125, 0.6212] | 0.5995 [0.5956, 0.6040] | 0.6033 [0.5998, 0.6081] | 0.5772 [0.5725, 0.5810] | 0.6261 [0.6215, 0.6314] | 0.6245 [0.6200, 0.6297] | 0.6214 [0.6173, 0.6259] | 0.7626 [0.7586, 0.7664] | 0.6245 [0.6200, 0.6297] | 0.6749 [0.6708, 0.6799] |
| LR-Both8 | 0.6251 [0.6214, 0.6300] | 0.5766 [0.5725, 0.5807] | 0.6169 [0.6133, 0.6222] | 0.5656 [0.5618, 0.5699] | 0.6286 [0.6242, 0.6337] | 0.6269 [0.6226, 0.6320] | 0.6267 [0.6232, 0.6322] | 0.7687 [0.7650, 0.7723] | 0.6269 [0.6226, 0.6320] | 0.6757 [0.6719, 0.6821] |
| LR-Both16 | 0.6183 [0.6142, 0.6243] | 0.5973 [0.5934, 0.6022] | 0.6135 [0.6096, 0.6199] | 0.5821 [0.5785, 0.5870] | 0.6260 [0.6215, 0.6316] | 0.6218 [0.6174, 0.6273] | 0.6309 [0.6271, 0.6368] | 0.7515 [0.7477, 0.7558] | 0.6218 [0.6174, 0.6273] | 0.6771 [0.6733, 0.6835] |
| LR-Both16-CV | 0.6137 [0.6098, 0.6195] | 0.5938 [0.5898, 0.5986] | 0.6132 [0.6092, 0.6199] | 0.5818 [0.5783, 0.5868] | 0.6213 [0.6168, 0.6271] | 0.6174 [0.6130, 0.6230] | 0.6309 [0.6270, 0.6371] | 0.7481 [0.7442, 0.7524] | 0.6174 [0.6130, 0.6230] | 0.6748 [0.6709, 0.6811] |
| Paul-style motif | 0.5963 [0.5924, 0.6007] | 0.5377 [0.5337, 0.5417] | 0.6101 [0.6064, 0.6144] | 0.5604 [0.5568, 0.5648] | 0.6026 [0.5984, 0.6073] | 0.6023 [0.5981, 0.6070] | 0.6217 [0.6182, 0.6265] | 0.7806 [0.7770, 0.7835] | 0.6023 [0.5981, 0.6070] | 0.6728 [0.6692, 0.6783] |

Study 42155283:

| model | Bond-AUROC N | Bond-AUROC C | Local-AUC N | Local-AUC C | Peptide-AUROC (both_bonds) | Length-matched AUC (both_bonds) | Near-miss AUC (both_bonds) | Peptide-AUROC (both_bonds_and_length) | Length-matched AUC (both_bonds_and_length) | Near-miss AUC (both_bonds_and_length) |
|---|---|---|---|---|---|---|---|---|---|---|
| LigTerm-GB | 0.6854 | 0.6578 | 0.6463 | 0.6050 | 0.7121 | 0.7091 | 0.6737 | 0.8141 | 0.7091 | 0.7123 |
| LR-Both8 | 0.6925 | 0.6175 | 0.6707 | 0.5878 | 0.7108 | 0.7093 | 0.6862 | 0.8151 | 0.7093 | 0.7220 |
| LR-Both16 | 0.6899 | 0.6553 | 0.6616 | 0.6136 | 0.7111 | 0.7060 | 0.6889 | 0.8116 | 0.7060 | 0.7230 |
| LR-Both16-CV | 0.6811 | 0.6481 | 0.6612 | 0.6130 | 0.7015 | 0.6971 | 0.6890 | 0.8075 | 0.6971 | 0.7192 |
| Paul-style motif | 0.6366 | 0.5545 | 0.6629 | 0.5807 | 0.6476 | 0.6472 | 0.6770 | 0.7766 | 0.6472 | 0.6924 |

Study 35051231:

| model | Bond-AUROC N | Bond-AUROC C | Local-AUC N | Local-AUC C | Peptide-AUROC (both_bonds) | Length-matched AUC (both_bonds) | Near-miss AUC (both_bonds) | Peptide-AUROC (both_bonds_and_length) | Length-matched AUC (both_bonds_and_length) | Near-miss AUC (both_bonds_and_length) |
|---|---|---|---|---|---|---|---|---|---|---|
| LigTerm-GB | 0.5469 | 0.5412 | 0.5602 | 0.5494 | 0.5400 | 0.5399 | 0.5690 | 0.7111 | 0.5399 | 0.6375 |
| LR-Both8 | 0.5577 | 0.5356 | 0.5631 | 0.5434 | 0.5463 | 0.5446 | 0.5672 | 0.7224 | 0.5446 | 0.6293 |
| LR-Both16 | 0.5468 | 0.5394 | 0.5654 | 0.5507 | 0.5410 | 0.5377 | 0.5729 | 0.6915 | 0.5377 | 0.6312 |
| LR-Both16-CV | 0.5464 | 0.5394 | 0.5651 | 0.5507 | 0.5412 | 0.5376 | 0.5728 | 0.6887 | 0.5376 | 0.6303 |
| Paul-style motif | 0.5559 | 0.5208 | 0.5572 | 0.5402 | 0.5575 | 0.5573 | 0.5665 | 0.7846 | 0.5573 | 0.6531 |

Descriptive paired differences (first minus second; pooled, with the interval; the per-study values follow):

| contrast | Bond-AUROC N | Bond-AUROC C | Local-AUC N | Local-AUC C | Length-matched AUC (both_bonds) | Near-miss AUC (both_bonds) | Peptide-AUROC (both_bonds) |
|---|---|---|---|---|---|---|---|
| LigTerm-GB minus LR-Both16 | -0.0021 [-0.0048, 0.0000] | 0.0022 [0.0001, 0.0042] | -0.0102 [-0.0142, -0.0074] | -0.0049 [-0.0093, -0.0019] | 0.0027 [0.0005, 0.0047] | -0.0095 [-0.0130, -0.0073] | 0.0000 [-0.0022, 0.0020] |
| &nbsp;&nbsp;42155283 | -0.0045 | 0.0026 | -0.0153 | -0.0086 | 0.0032 | -0.0151 | 0.0010 |
| &nbsp;&nbsp;35051231 | 0.0002 | 0.0018 | -0.0051 | -0.0012 | 0.0023 | -0.0039 | -0.0010 |
| LigTerm-GB minus LR-Both16-CV | 0.0025 [0.0001, 0.0044] | 0.0057 [0.0037, 0.0078] | -0.0099 [-0.0141, -0.0069] | -0.0046 [-0.0092, -0.0015] | 0.0072 [0.0049, 0.0090] | -0.0096 [-0.0131, -0.0071] | 0.0047 [0.0025, 0.0066] |
| &nbsp;&nbsp;42155283 | 0.0044 | 0.0097 | -0.0150 | -0.0080 | 0.0120 | -0.0152 | 0.0107 |
| &nbsp;&nbsp;35051231 | 0.0006 | 0.0018 | -0.0049 | -0.0012 | 0.0023 | -0.0039 | -0.0012 |
| LR-Both16-CV minus LR-Both16 | -0.0046 [-0.0052, -0.0041] | -0.0036 [-0.0039, -0.0032] | -0.0003 [-0.0010, 0.0007] | -0.0003 [-0.0009, 0.0005] | -0.0045 [-0.0048, -0.0041] | 0.0000 [-0.0004, 0.0006] | -0.0047 [-0.0050, -0.0043] |
| &nbsp;&nbsp;42155283 | -0.0088 | -0.0071 | -0.0003 | -0.0006 | -0.0089 | 0.0001 | -0.0096 |
| &nbsp;&nbsp;35051231 | -0.0004 | 0.0000 | -0.0002 | 0.0000 | -0.0000 | -0.0000 | 0.0002 |
| LR-Both16 minus LR-Both8 | -0.0067 [-0.0087, -0.0043] | 0.0208 [0.0183, 0.0238] | -0.0034 [-0.0057, 0.0000] | 0.0165 [0.0140, 0.0196] | -0.0051 [-0.0069, -0.0030] | 0.0042 [0.0024, 0.0062] | -0.0025 [-0.0045, -0.0004] |
| &nbsp;&nbsp;42155283 | -0.0026 | 0.0377 | -0.0091 | 0.0258 | -0.0033 | 0.0027 | 0.0003 |
| &nbsp;&nbsp;35051231 | -0.0109 | 0.0038 | 0.0023 | 0.0073 | -0.0069 | 0.0057 | -0.0053 |
| LR-Both16 minus Paul-style motif | 0.0221 [0.0186, 0.0268] | 0.0597 [0.0562, 0.0638] | 0.0034 [0.0001, 0.0084] | 0.0217 [0.0189, 0.0253] | 0.0196 [0.0158, 0.0238] | 0.0092 [0.0069, 0.0126] | 0.0235 [0.0197, 0.0277] |
| &nbsp;&nbsp;42155283 | 0.0533 | 0.1007 | -0.0014 | 0.0328 | 0.0588 | 0.0119 | 0.0635 |
| &nbsp;&nbsp;35051231 | -0.0092 | 0.0186 | 0.0081 | 0.0105 | -0.0197 | 0.0064 | -0.0166 |

Selection (LR-Both16-CV: penalty by the rule on the training folds; LigTerm-GB: boosting iterations by the rule on the training folds): 42155283: N C=0.01, 50 it; C C=0.01, 50 it; 35051231: N C=0.3, 100 it; C C=0.1, 100 it

## Regime identity_50_grouping

| study | training units | proteins | training positives N / C | evaluation proteins (bond level N / C) | candidate positives |
|---|---|---|---|---|---|
| 42155283 | 27952 | 7102 | 22506 / 22642 | 4600 / 4455 | 33100 |
| 35051231 | 39480 | 5046 | 24344 / 24389 | 7668 / 7666 | 24163 |

Pooled values with 95% intervals (study values below):

| model | Bond-AUROC N | Bond-AUROC C | Local-AUC N | Local-AUC C | Peptide-AUROC (both_bonds) | Length-matched AUC (both_bonds) | Near-miss AUC (both_bonds) | Peptide-AUROC (both_bonds_and_length) | Length-matched AUC (both_bonds_and_length) | Near-miss AUC (both_bonds_and_length) |
|---|---|---|---|---|---|---|---|---|---|---|
| LigTerm-GB | 0.6257 [0.6220, 0.6301] | 0.6058 [0.6013, 0.6100] | 0.6176 [0.6137, 0.6218] | 0.5832 [0.5790, 0.5872] | 0.6350 [0.6305, 0.6398] | 0.6327 [0.6283, 0.6374] | 0.6353 [0.6313, 0.6393] | 0.7614 [0.7579, 0.7649] | 0.6327 [0.6283, 0.6374] | 0.6797 [0.6754, 0.6838] |
| LR-Both8 | 0.6286 [0.6248, 0.6326] | 0.5752 [0.5712, 0.5792] | 0.6251 [0.6212, 0.6290] | 0.5678 [0.5638, 0.5715] | 0.6320 [0.6277, 0.6362] | 0.6304 [0.6261, 0.6346] | 0.6347 [0.6308, 0.6386] | 0.7677 [0.7643, 0.7710] | 0.6304 [0.6261, 0.6346] | 0.6791 [0.6748, 0.6834] |
| LR-Both16 | 0.6233 [0.6194, 0.6275] | 0.6007 [0.5964, 0.6049] | 0.6219 [0.6180, 0.6261] | 0.5865 [0.5824, 0.5905] | 0.6319 [0.6273, 0.6367] | 0.6273 [0.6228, 0.6320] | 0.6408 [0.6369, 0.6450] | 0.7535 [0.7500, 0.7571] | 0.6273 [0.6228, 0.6320] | 0.6820 [0.6775, 0.6864] |
| LR-Both16-CV | 0.6209 [0.6170, 0.6252] | 0.5983 [0.5940, 0.6024] | 0.6221 [0.6181, 0.6264] | 0.5857 [0.5817, 0.5897] | 0.6290 [0.6243, 0.6339] | 0.6245 [0.6199, 0.6292] | 0.6402 [0.6363, 0.6445] | 0.7511 [0.7474, 0.7547] | 0.6245 [0.6199, 0.6292] | 0.6800 [0.6755, 0.6843] |
| Paul-style motif | 0.5960 [0.5918, 0.6003] | 0.5335 [0.5295, 0.5376] | 0.6124 [0.6082, 0.6166] | 0.5589 [0.5550, 0.5625] | 0.6001 [0.5957, 0.6045] | 0.5997 [0.5953, 0.6041] | 0.6230 [0.6193, 0.6271] | 0.7769 [0.7739, 0.7801] | 0.5997 [0.5953, 0.6041] | 0.6716 [0.6674, 0.6759] |

Study 42155283:

| model | Bond-AUROC N | Bond-AUROC C | Local-AUC N | Local-AUC C | Peptide-AUROC (both_bonds) | Length-matched AUC (both_bonds) | Near-miss AUC (both_bonds) | Peptide-AUROC (both_bonds_and_length) | Length-matched AUC (both_bonds_and_length) | Near-miss AUC (both_bonds_and_length) |
|---|---|---|---|---|---|---|---|---|---|---|
| LigTerm-GB | 0.7052 | 0.6713 | 0.6685 | 0.6171 | 0.7313 | 0.7272 | 0.6987 | 0.8176 | 0.7272 | 0.7223 |
| LR-Both8 | 0.7016 | 0.6181 | 0.6853 | 0.5913 | 0.7204 | 0.7190 | 0.7004 | 0.8151 | 0.7190 | 0.7276 |
| LR-Both16 | 0.7029 | 0.6644 | 0.6783 | 0.6212 | 0.7260 | 0.7202 | 0.7072 | 0.8165 | 0.7202 | 0.7305 |
| LR-Both16-CV | 0.6983 | 0.6594 | 0.6783 | 0.6196 | 0.7201 | 0.7147 | 0.7059 | 0.8130 | 0.7147 | 0.7270 |
| Paul-style motif | 0.6374 | 0.5500 | 0.6663 | 0.5797 | 0.6462 | 0.6458 | 0.6806 | 0.7709 | 0.6458 | 0.6908 |

Study 35051231:

| model | Bond-AUROC N | Bond-AUROC C | Local-AUC N | Local-AUC C | Peptide-AUROC (both_bonds) | Length-matched AUC (both_bonds) | Near-miss AUC (both_bonds) | Peptide-AUROC (both_bonds_and_length) | Length-matched AUC (both_bonds_and_length) | Near-miss AUC (both_bonds_and_length) |
|---|---|---|---|---|---|---|---|---|---|---|
| LigTerm-GB | 0.5462 | 0.5402 | 0.5666 | 0.5492 | 0.5387 | 0.5382 | 0.5719 | 0.7053 | 0.5382 | 0.6372 |
| LR-Both8 | 0.5557 | 0.5324 | 0.5649 | 0.5443 | 0.5435 | 0.5417 | 0.5689 | 0.7203 | 0.5417 | 0.6307 |
| LR-Both16 | 0.5437 | 0.5371 | 0.5656 | 0.5518 | 0.5378 | 0.5343 | 0.5744 | 0.6905 | 0.5343 | 0.6335 |
| LR-Both16-CV | 0.5436 | 0.5371 | 0.5658 | 0.5518 | 0.5380 | 0.5343 | 0.5746 | 0.6891 | 0.5343 | 0.6331 |
| Paul-style motif | 0.5547 | 0.5170 | 0.5584 | 0.5381 | 0.5539 | 0.5537 | 0.5654 | 0.7830 | 0.5537 | 0.6524 |

Descriptive paired differences (first minus second; pooled, with the interval; the per-study values follow):

| contrast | Bond-AUROC N | Bond-AUROC C | Local-AUC N | Local-AUC C | Length-matched AUC (both_bonds) | Near-miss AUC (both_bonds) | Peptide-AUROC (both_bonds) |
|---|---|---|---|---|---|---|---|
| LigTerm-GB minus LR-Both16 | 0.0024 [0.0005, 0.0044] | 0.0051 [0.0031, 0.0070] | -0.0044 [-0.0068, -0.0017] | -0.0033 [-0.0059, -0.0006] | 0.0054 [0.0036, 0.0073] | -0.0055 [-0.0077, -0.0033] | 0.0031 [0.0012, 0.0050] |
| &nbsp;&nbsp;42155283 | 0.0023 | 0.0069 | -0.0098 | -0.0041 | 0.0070 | -0.0085 | 0.0053 |
| &nbsp;&nbsp;35051231 | 0.0025 | 0.0032 | 0.0010 | -0.0026 | 0.0039 | -0.0025 | 0.0009 |
| LigTerm-GB minus LR-Both16-CV | 0.0048 [0.0029, 0.0068] | 0.0075 [0.0056, 0.0094] | -0.0045 [-0.0069, -0.0018] | -0.0025 [-0.0052, 0.0002] | 0.0082 [0.0063, 0.0100] | -0.0049 [-0.0072, -0.0028] | 0.0060 [0.0040, 0.0078] |
| &nbsp;&nbsp;42155283 | 0.0070 | 0.0119 | -0.0098 | -0.0025 | 0.0126 | -0.0071 | 0.0113 |
| &nbsp;&nbsp;35051231 | 0.0026 | 0.0032 | 0.0008 | -0.0026 | 0.0039 | -0.0027 | 0.0007 |
| LR-Both16-CV minus LR-Both16 | -0.0024 [-0.0027, -0.0021] | -0.0025 [-0.0027, -0.0023] | 0.0001 [-0.0005, 0.0008] | -0.0008 [-0.0013, -0.0003] | -0.0028 [-0.0030, -0.0025] | -0.0006 [-0.0008, -0.0002] | -0.0029 [-0.0031, -0.0026] |
| &nbsp;&nbsp;42155283 | -0.0046 | -0.0050 | 0.0000 | -0.0016 | -0.0056 | -0.0013 | -0.0059 |
| &nbsp;&nbsp;35051231 | -0.0001 | 0.0000 | 0.0002 | 0.0000 | 0.0000 | 0.0002 | 0.0002 |
| LR-Both16 minus LR-Both8 | -0.0053 [-0.0074, -0.0031] | 0.0255 [0.0228, 0.0283] | -0.0031 [-0.0054, -0.0009] | 0.0187 [0.0159, 0.0214] | -0.0031 [-0.0050, -0.0012] | 0.0061 [0.0043, 0.0081] | -0.0001 [-0.0020, 0.0019] |
| &nbsp;&nbsp;42155283 | 0.0013 | 0.0463 | -0.0070 | 0.0299 | 0.0012 | 0.0068 | 0.0056 |
| &nbsp;&nbsp;35051231 | -0.0119 | 0.0047 | 0.0008 | 0.0075 | -0.0074 | 0.0055 | -0.0057 |
| LR-Both16 minus Paul-style motif | 0.0273 [0.0236, 0.0311] | 0.0673 [0.0631, 0.0711] | 0.0096 [0.0065, 0.0127] | 0.0276 [0.0244, 0.0308] | 0.0276 [0.0237, 0.0316] | 0.0178 [0.0153, 0.0204] | 0.0318 [0.0279, 0.0359] |
| &nbsp;&nbsp;42155283 | 0.0655 | 0.1144 | 0.0119 | 0.0415 | 0.0745 | 0.0266 | 0.0798 |
| &nbsp;&nbsp;35051231 | -0.0109 | 0.0201 | 0.0072 | 0.0136 | -0.0193 | 0.0089 | -0.0162 |

Selection (LR-Both16-CV: penalty by the rule on the training folds; LigTerm-GB: boosting iterations by the rule on the training folds): 42155283: N C=0.01, 50 it; C C=0.01, 50 it; 35051231: N C=0.3, 100 it; C C=0.1, 200 it

## Development medians beside the historical-test values (19 unseen development studies; the group-disjoint regime of every development run)

| model | Bond-AUROC N | Bond-AUROC C | Local-AUC N | Local-AUC C | Length-matched AUC (both_bonds) | Near-miss AUC (both_bonds) | Peptide-AUROC (both_bonds) |
|---|---|---|---|---|---|---|---|
| LigTerm-GB | 0.7755 | 0.7606 | 0.7485 | 0.7164 | 0.8066 | 0.7974 | 0.8134 |
| LR-Both8 | 0.7257 | 0.6857 | 0.7280 | 0.6836 | 0.7862 | 0.7706 | 0.7866 |
| LR-Both16 | 0.7451 | 0.7324 | 0.7375 | 0.7250 | 0.7816 | 0.7966 | 0.7886 |
| LR-Both16-CV | 0.7451 | 0.7326 | 0.7375 | 0.7250 | 0.7810 | 0.7966 | 0.7889 |
| Paul-style motif | 0.6625 | 0.6432 | 0.6986 | 0.6730 | 0.7276 | 0.7433 | 0.7287 |

## Consistency checks of the run records (before the aggregation; a failure stops it)

all passed; notes (not fatal): 8; bootstrap draws that were not finite (left out of the percentiles and counted): 0
- note: 42155283 identity_50_grouping N LR-Both16-CV: the penalty C = 0.01 is at the edge of the grid
- note: 42155283 identity_50_grouping N LigTerm-GB: 50 iterations is at the edge of the grid
- note: 42155283 identity_50_grouping C LR-Both16-CV: the penalty C = 0.01 is at the edge of the grid
- note: 42155283 identity_50_grouping C LigTerm-GB: 50 iterations is at the edge of the grid
- note: 42155283 lab_excluded N LR-Both16-CV: the penalty C = 0.01 is at the edge of the grid
- note: 42155283 lab_excluded N LigTerm-GB: 50 iterations is at the edge of the grid
- note: 42155283 lab_excluded C LR-Both16-CV: the penalty C = 0.01 is at the edge of the grid
- note: 42155283 lab_excluded C LigTerm-GB: 50 iterations is at the edge of the grid
