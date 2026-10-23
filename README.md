# 21 點雙層 DQN：出牌策略與自主計數概念

這個專案以兩個完全獨立的神經網路處理不同時間點的決策：

1. `BetAgent` 在發牌前只看 11 維牌靴組成。
2. `PlayAgent` 在發牌後看 16 維玩家／莊家／牌靴狀態。

訓練嚴格分成兩階段。先以平注訓練出牌 Double DQN，再凍結出牌網路，以 1 億回合訓練下注網路。Oracle 和 Hi-Lo 只存在於評估程式，不會進入訓練。

## 規則與重要實作

- 預設 2 副牌、S17、DAS、最多 4 手、3:2 Blackjack、American peek。
- cut card 只在新回合開始前檢查；回合中不因跌破 15 張洗牌。
- 若單一極端回合真的抽乾牌靴，才重洗棄牌，且排除仍在本回合桌面上的牌；次數會記錄在 log。
- 分牌採 lazy deal：第一手完成後，第二手才收到第二張牌。
- 莊家 A／10 明牌先 peek。兩家天然 Blackjack 為 push；只有玩家天然 Blackjack 才得 `+1.5 unit`。
- 出牌 reward 用 unit；下注 reward 用 `units × bet / 1000`，因此三個下注尺度為 `{0.1, 0.5, 1.0}`。
- `BetAgent` 的 `γ=0`。下注不會影響牌堆轉移，所以這個 MDP 精確退化成 contextual bandit：

  `Q(s,a) = E[r | s,a]`

這是問題本身的結構，不是省略時間差分學習。線性期望值目標預期產生 bang-bang 策略：正 EV 押 1000，非正 EV 押 100；500 很少成為唯一最優解。

## 安裝與測試

建議使用有 CUDA 的 PyTorch 環境：

```bash
cd blackjack_dqn
python -m pip install -r requirements.txt
pytest -q
```

一鍵 smoke test 會以極小資料量跑過 Phase 1、Phase 2 與完整評估管線；它只驗證程式整合，不代表模型收斂：

```bash
python main.py --output outputs/smoke smoke
```

## 正式訓練

Phase 1 預設 2,000 萬回合：

```bash
python main.py --output outputs/full train-play
```

Phase 2 預設 1 億回合，且一定要載入已完成的出牌模型：

```bash
python main.py --output outputs/full train-bet \
  --play outputs/full/play_final.pt
```

可用 `--resume` 接續對應階段的 checkpoint。正式評估：

```bash
python main.py --output outputs/full evaluate \
  --play outputs/full/play_final.pt \
  --bet outputs/full/bet_final.pt \
  --visits outputs/full/play_visits.npz
```

## 評估產物

`outputs/full/evaluation/` 會包含：

- 硬牌、軟牌、分牌在 TC≈0 與 TC≈+5 的 CSV 與 heatmap
- 中性／高真數 diff 矩陣
- 每格鄰域 visit count 與低信心灰階
- `weights_vs_hilo.png`
- `bet_ramp.png`
- `cumulative_regret.png`
- `oracle_samples.csv`
- 至少 50 萬回合的獲利平均、標準誤與 95% CI
- 彙整所有方法和數字的 `report.json`

策略表不是把任意 total 塞進網路。每格都有 canonical 手牌，會先從牌靴扣除玩家牌和莊家明牌，再調整牌靴組成，最後重新驗證 TC。例：`hard 16 v 10` 使用 `10+6` 並扣除兩張 10、一張 6。

Visit count 是從訓練期間抽樣保存的原始牌值張數，直到 `evaluate.py` 才換算 TC。定義為「策略格 × TC 區間 × 穿透率區間」的估計訪問量；預設 TC 視窗為 ±0.5。若高真數樣本中位數不足，會自動放寬為 ±1.0，實際寬度寫入報告。

## 專案檔案

- `config.py`：規則與超參數 dataclass
- `env.py`：向量化牌靴、peek、hand queue、lazy split、遮罩
- `agents.py`：兩個網路、Double DQN、replay buffer
- `train.py`：嚴格分離的兩階段訓練
- `evaluate.py`：自洽策略表、計數概念、regret、CI
- `oracle.py`：只供評估的基本策略 Monte Carlo EV
- `visit_stats.py`：不含任何人工計數公式的原始 visit 抽樣
- `main.py`：CLI
- `tests/`：規則、轉移、遮罩、模組邊界測試

## 計算量提醒

預設正式設定是課程實驗等級，CPU 可能需要很長時間。Smoke test 通過只表示規則與資料流可執行；是否學出基本策略與計數概念，必須以正式訓練曲線、策略表、visit count、oracle regret 和信賴區間判斷，不能用短跑結果下結論。
