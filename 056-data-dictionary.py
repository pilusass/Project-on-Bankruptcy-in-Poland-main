#!/usr/bin/env python
# coding: utf-8

# <font size="+3"><strong>5.6. Data Dictionary</strong></font>

# # Poland Bankruptcy Data

# Below is a summary of the features from the [Poland bankruptcy dataset](https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data). 

# | feature    | description                                                                                                         |  
# | :--------- | :------------------------------------------------------------------------------------------------------------------ |  
# | **feat_1**  | net profit / total assets                                                                                           |  
# | **feat_2**  | total liabilities / total assets                                                                                    |  
# | **feat_3**  | working capital / total assets                                                                                      |  
# | **feat_4**  | current assets / short-term liabilities                                                                             |  
# | **feat_5**  | [(cash + short-term securities + receivables - short-term liabilities) / (operating expenses - depreciation)] * 365 |  
# | **feat_6**  | retained earnings / total assets                                                                                    |  
# | **feat_7**  | EBIT / total assets                                                                                                 |  
# | **feat_8**  | book value of equity / total liabilities                                                                            |  
# | **feat_9**  | sales / total assets                                                                                                |  
# | **feat_10** | equity / total assets                                                                                               |  
# | **feat_11** | (gross profit + extraordinary items + financial expenses) / total assets                                            |  
# | **feat_12** | gross profit / short-term liabilities                                                                               |  
# | **feat_13** | (gross profit + depreciation) / sales                                                                               |  
# | **feat_14** | (gross profit + interest) / total assets                                                                            |  
# | **feat_15** | (total liabilities * 365) / (gross profit + depreciation)                                                           |  
# | **feat_16** | (gross profit + depreciation) / total liabilities                                                                   |  
# | **feat_17** | total assets / total liabilities                                                                                    |  
# | **feat_18** | gross profit / total assets                                                                                         |  
# | **feat_19** | gross profit / sales                                                                                                |  
# | **feat_20** | (inventory * 365) / sales                                                                                           |  
# | **feat_21** | sales (n) / sales (n-1)                                                                                             |  
# | **feat_22** | profit on operating activities / total assets                                                                       |  
# | **feat_23** | net profit / sales                                                                                                  |  
# | **feat_24** | gross profit (in 3 years) / total assets                                                                            |  
# | **feat_25** | (equity - share capital) / total assets                                                                             |  
# | **feat_26** | (net profit + depreciation) / total liabilities                                                                     |  
# | **feat_27** | profit on operating activities / financial expenses                                                                 |  
# | **feat_28** | working capital / fixed assets                                                                                      |  
# | **feat_29** | logarithm of total assets                                                                                           |  
# | **feat_30** | (total liabilities - cash) / sales                                                                                  |  
# | **feat_31** | (gross profit + interest) / sales                                                                                   |  
# | **feat_32** | (current liabilities * 365) / cost of products sold                                                                 |  
# | **feat_33** | operating expenses / short-term liabilities                                                                         |  
# | **feat_34** | operating expenses / total liabilities                                                                              |  
# | **feat_35** | profit on sales / total assets                                                                                      |  
# | **feat_36** | total sales / total assets                                                                                          |  
# | **feat_37** | (current assets - inventories) / long-term liabilities                                                              |  
# | **feat_38** | constant capital / total assets                                                                                     |  
# | **feat_39** | profit on sales / sales                                                                                             |  
# | **feat_40** | (current assets - inventory - receivables) / short-term liabilities                                                 |  
# | **feat_41** | total liabilities / ((profit on operating activities + depreciation) * (12/365))                                    |  
# | **feat_42** | profit on operating activities / sales                                                                              |  
# | **feat_43** | rotation receivables + inventory turnover in days                                                                   |  
# | **feat_44** | (receivables * 365) / sales                                                                                         |  
# | **feat_45** | net profit / inventory                                                                                              |  
# | **feat_46** | (current assets - inventory) / short-term liabilities                                                               |  
# | **feat_47** | (inventory * 365) / cost of products sold                                                                           |  
# | **feat_48** | EBITDA (profit on operating activities - depreciation) / total assets                                               |  
# | **feat_49** | EBITDA (profit on operating activities - depreciation) / sales                                                      |  
# | **feat_50** | current assets / total liabilities                                                                                  |  
# | **feat_51** | short-term liabilities / total assets                                                                               |  
# | **feat_52** | (short-term liabilities * 365) / cost of products sold)                                                             |  
# | **feat_53** | equity / fixed assets                                                                                               |  
# | **feat_54** | constant capital / fixed assets                                                                                     |  
# | **feat_55** | working capital                                                                                                     |  
# | **feat_56** | (sales - cost of products sold) / sales                                                                             |  
# | **feat_57** | (current assets - inventory - short-term liabilities) / (sales - gross profit - depreciation)                       |  
# | **feat_58** | total costs /total sales                                                                                            |  
# | **feat_59** | long-term liabilities / equity                                                                                      |  
# | **feat_60** | sales / inventory                                                                                                   |  
# | **feat_61** | sales / receivables                                                                                                 |  
# | **feat_62** | (short-term liabilities *365) / sales                                                                               |  
# | **feat_63** | sales / short-term liabilities                                                                                      |  
# | **feat_64** | sales / fixed assets                                                                                                |  
# | **bankrupt** | Whether company went bankrupt at end of forecasting period (2013) |

# # Taiwan Bankruptcy Dataset

# Below is a summary of the features from the [Taiwan bankruptcy dataset](https://archive.ics.uci.edu/ml/datasets/Taiwanese+Bankruptcy+Prediction). 
# 
# **Note:** All of the variables have been normalized into the range from 0 to 1.

# | feature      | description                                             |
# | :----------- | :------------------------------------------------------ |
# | **bankrupt** | Whether or not company has gone bankrupt                |
# | **feat_1**   | ROA(C) before interest and depreciation before interest |
# | **feat_2**   | ROA(A) before interest and % after tax                  |
# | **feat_3**   | ROA(B) before interest and depreciation after tax       |
# | **feat_4**   | Operating Gross Margin                                  |
# | **feat_5**   | Realized Sales Gross Margin                             |
# | **feat_6**   | Operating Profit Rate                                   |
# | **feat_7**   | Pre-tax net Interest Rate                               |
# | **feat_8**   | After-tax net Interest Rate                             |
# | **feat_9**   | Non-industry income and expenditure/revenue             |
# | **feat_10**  | Continuous interest rate (after tax)                    |
# | **feat_11**  | Operating Expense Rate                                  |
# | **feat_12**  | Research and development expense rate                   |
# | **feat_13**  | Cash flow rate                                          |
# | **feat_14**  | Interest-bearing debt interest rate                     |
# | **feat_15**  | Tax rate (A)                                            |
# | **feat_16**  | Net Value Per Share (B)                                 |
# | **feat_17**  | Net Value Per Share (A)                                 |
# | **feat_18**  | Net Value Per Share (C)                                 |
# | **feat_19**  | Persistent EPS in the Last Four Seasons                 |
# | **feat_20**  | Cash Flow Per Share                                     |
# | **feat_21**  | Revenue Per Share (Yuan ¥)                              |
# | **feat_22**  | Operating Profit Per Share (Yuan ¥)                     |
# | **feat_23**  | Per Share Net profit before tax (Yuan ¥)                |
# | **feat_24**  | Realized Sales Gross Profit Growth Rate                 |
# | **feat_25**  | Operating Profit Growth Rate                            |
# | **feat_26**  | After-tax Net Profit Growth Rate                        |
# | **feat_27**  | Regular Net Profit Growth Rate                          |
# | **feat_28**  | Continuous Net Profit Growth Rate                       |
# | **feat_29**  | Total Asset Growth Rate                                 |
# | **feat_30**  | Net Value Growth Rate                                   |
# | **feat_31**  | Total Asset Return Growth Rate Ratio                    |
# | **feat_32**  | Cash Reinvestment %                                     |
# | **feat_33**  | Current Ratio                                           |
# | **feat_34**  | Quick Ratio                                             |
# | **feat_35**  | Interest Expense Ratio                                  |
# | **feat_36**  | Total debt/Total net worth                              |
# | **feat_37**  | Debt ratio %                                            |
# | **feat_38**  | Net worth/Assets                                        |
# | **feat_39**  | Long-term fund suitability ratio (A)                    |
# | **feat_40**  | Borrowing dependency                                    |
# | **feat_41**  | Contingent liabilities/Net worth                        |
# | **feat_42**  | Operating profit/Paid-in capital                        |
# | **feat_43**  | Net profit before tax/Paid-in capital                   |
# | **feat_44**  | Inventory and accounts receivable/Net value             |
# | **feat_45**  | Total Asset Turnover                                    |
# | **feat_46**  | Accounts Receivable Turnover                            |
# | **feat_47**  | Average Collection Days                                 |
# | **feat_48**  | Inventory Turnover Rate (times)                         |
# | **feat_49**  | Fixed Assets Turnover Frequency                         |
# | **feat_50**  | Net Worth Turnover Rate (times)                         |
# | **feat_51**  | Revenue per person                                      |
# | **feat_52**  | Operating profit per person                             |
# | **feat_53**  | Allocation rate per person                              |
# | **feat_54**  | Working Capital to Total Assets                         |
# | **feat_55**  | Quick Assets/Total Assets                               |
# | **feat_56**  | Current Assets/Total Assets                             |
# | **feat_57**  | Cash/Total Assets                                       |
# | **feat_58**  | Quick Assets/Current Liability                          |
# | **feat_59**  | Cash/Current Liability                                  |
# | **feat_60**  | Current Liability to Assets                             |
# | **feat_61**  | Operating Funds to Liability                            |
# | **feat_62**  | Inventory/Working Capital                               |
# | **feat_63**  | Inventory/Current Liability                             |
# | **feat_64**  | Current Liabilities/Liability                           |
# | **feat_65**  | Working Capital/Equity                                  |
# | **feat_66**  | Current Liabilities/Equity                              |
# | **feat_67**  | Long-term Liability to Current Assets                   |
# | **feat_68**  | Retained Earnings to Total Assets                       |
# | **feat_69**  | Total income/Total expense                              |
# | **feat_70**  | Total expense/Assets                                    |
# | **feat_71**  | Current Asset Turnover Rate                             |
# | **feat_72**  | Quick Asset Turnover Rate                               |
# | **feat_73**  | Working Capital Turnover Rate                           |
# | **feat_74**  | Cash Turnover Rate                                      |
# | **feat_75**  | Cash Flow to Sales                                      |
# | **feat_76**  | Fixed Assets to Assets                                  |
# | **feat_77**  | Current Liability to Liability                          |
# | **feat_78**  | Current Liability to Equity                             |
# | **feat_79**  | Equity to Long-term Liability                           |
# | **feat_80**  | Cash Flow to Total Assets                               |
# | **feat_81**  | Cash Flow to Liability                                  |
# | **feat_82**  | CFO to Assets                                           |
# | **feat_83**  | Cash Flow to Equity                                     |
# | **feat_84**  | Current Liability to Current Assets                     |
# | **feat_85**  | Liability-Assets Flag                                   |
# | **feat_86**  | Net Income to Total Assets                              |
# | **feat_87**  | Total assets to GNP price                               |
# | **feat_88**  | No-credit Interval                                      |
# | **feat_89**  | Gross Profit to Sales                                   |
# | **feat_90**  | Net Income to Stockholder's Equity                      |
# | **feat_91**  | Liability to Equity                                     |
# | **feat_92**  | Degree of Financial Leverage (DFL)                      |
# | **feat_93**  | Interest Coverage Ratio (Interest expense to EBIT)      |
# | **feat_94**  | Net Income Flag                                         |
# | **feat_95**  | Equity to Liability                                     |
# 

# ---
# Copyright © 2022 WorldQuant University. This
# content is licensed solely for personal use. Redistribution or
# publication of this material is strictly prohibited.
# 
