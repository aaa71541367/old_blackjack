# 期末報告方法與結果骨架

## 研究問題

驗證兩個只接收原始牌靴組成的獨立 DQN，是否能分別學出 21 點基本出牌策略及與 Hi-Lo 類似的下注訊號。Hi-Lo 不屬於訓練輸入，只在事後分析使用。

## 方法摘要

- 出牌層：16→256→256→4 Double DQN，`γ=1`，合法動作同時遮罩於行為選擇與 target argmax。
- 下注層：11→128→128→3 DQN，`γ=0`，reward 為 `units × bet / 1000`。
- 訓練順序：先訓練出牌層至少 2,000 萬回合；凍結後，再訓練下注層至少 1 億回合。
- 下注層是 contextual bandit，因下注額不影響下一個牌靴狀態，Bellman 方程降為 `Q(s,a)=E[r|s,a]`。
- 分牌軌跡以 hand queue 串接；split reward 為 0，各子手最終動作承接自己的結果，最後一手才 `done=True`。
- 莊家結束後才回填延遲 reward，整個回合完成後才寫入 replay buffer。

## 規則控制

使用 2 副牌、S17、DAS、最多 4 手、3:2、American peek。分牌採 lazy deal，避免第二手未來牌洩漏。cut card 只在回合邊界生效；抽乾 fallback 會被單獨記錄。

## 必填結果

正式執行後，將 `evaluation/report.json` 的數字及下列圖片放入報告：

1. Phase 1 loss／EV 曲線。
2. 中性牌靴硬牌、軟牌、分牌策略表。
3. TC≈+5 策略表及 diff，對照 Illustrious 18。
4. 每格 visit count；低於 1,000 的格子不得作強結論。
5. `Q(1000)-Q(100)` 線性係數與 Hi-Lo 權重關係。
6. 下注動作對 true count 的 bet ramp。
7. 對 Monte Carlo oracle 的 cumulative regret。
8. 依 `|TC|` 分層的決策準確率。
9. 至少 50 萬回合的平均每手損益與 95% CI。

## 結論判準

不能只用總獲利判定成功。出牌策略須在高訪問格接近基本策略；下注網路的組成權重須呈現與高低牌價值一致的結構；bet ramp、oracle regret 與信賴區間必須共同支持結論。500 注碼很少被選中屬於線性期望值目標的合理 bang-bang 解，不代表訓練失敗。
