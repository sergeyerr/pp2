| Setup | Loss | RMSE |
|-------|------|------|
| prott5+esm2 (3584 dim) + simple MLP (lr=1e-4, weight_decay=0, batch_size=64, patience=5)  | 0.2604     | 2.5213     |
| prott5+esm2+prostt5 (4608 dim) + simple MLP (lr=1e-4, weight_decay=0, batch_size=64, patience=5)      | 0.2464     | 2.4422     |
| prott5+esm2+prostt5 (4608 dim) + simple MLP (lr=1e-4, weight_decay=1e-2, batch_size=64, patience=5)      |   0.2853   | 2.6656     |
| prott5+esm2+prostt5 (4608 dim) + simple MLP (lr=1e-4, weight_decay=1e-3, batch_size=64, patience=5)      | 0.2648     | 2.5535     |
| prott5+esm2+prostt5+prott5_protein (5632 dim) + simple MLP (lr=1e-4, weight_decay=1e-3, batch_size=64, patience=5)      | 0.2629     | 2.5432     |
| prott5+esm2+prostt5+prott5_protein (5632 dim) + simple MLP (lr=1e-4, weight_decay=0, batch_size=64, patience=5)      | 0.2430     | 2.4204     |
| prott5+esm2+prostt5+prott5_protein (5632 dim) + LA + Simple MLP (lr=1e-5, weight_decay=0, batch_size=64, patience=10)      | 0.2423     | 2.4180     |
| prott5+esm2+prostt5+prott5_protein (5632 dim) + LA + Simple MLP (lr=1e-4, weight_decay=0, batch_size=64, patience=10)      | 0.2374     | 2.3863     |
| prott5+esm2+prostt5+prott5_protein (5632 dim) + LA + Simple MLP (lr=5-e5, weight_decay=0, batch_size=64, patience=10)      | 0.2407     | 2.4092     |
| prott5+esm2+prostt5+prott5_protein+prostt5_protein      |      |      |
| prott5      |      |      |
| prostt5      |      |      |
| esm2      |      |      |
|       |      |      |
|       |      |      |
|       |      |      |
|       |      |      |

