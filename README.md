# Mamba Chess + MCTS

研究棋盤序列表示、policy/value 預測與 MCTS 搜尋的 Python 原型，包含 BoardMamba、CNN baseline、合法行動遮罩、自我對弈和 Stockfish 評估入口。

這份作品可檢查從棋盤編碼到搜尋決策的連接方式。既有小型實驗中，CNN 的部分指標較好，尚未建立 Mamba 優勢或可靠的棋力等級。

## 從哪裡看

| 程式 | 負責內容 |
|---|---|
| [棋盤與行動編碼](src/chess_env/) | 70-token 序列、CNN planes、legal mask、PGN 解析與整局資料切分 |
| [模型](src/models/) | BoardMamba、2D positional encoding、CNN 與 policy/value heads |
| [搜尋](src/search/) | PUCT、MCTS、快取、自我對弈 |
| [訓練](src/train/) | policy、value、joint、自我對弈與棋局復盤訓練 |
| [評估](src/eval/) | head-to-head、Stockfish 比較、信賴區間與耗時 |
| [測試](tests/) | 行動映射、合法走法、棋盤表示、PGN、value 符號與終局 |

## 安裝與基本驗證

Python 3.10+。先依硬體安裝 PyTorch，再安裝其餘依賴：

```bash
python -m venv .venv
# 啟用 .venv
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
```

2026-10-09 公開版驗收：Windows、Python 3.12，**18 項測試通過**。這次測試不需要下載棋譜、Stockfish 或模型權重，也不代表真實棋力驗證。

## 真正的 Mamba 與 fallback

`mamba-ssm` 使用 CUDA 擴充。Linux／WSL 的歷史設定列於 [environment_mamba_real.yml](environment_mamba_real.yml)，安裝方式以 [Mamba 官方說明](https://github.com/state-spaces/mamba)為準。

當 `mamba-ssm` 無法載入時，現行程式會選用 GRU fallback，名稱明確標為 `MambaFallback`。開始任何 Mamba 實驗前，務必檢查：

```bash
python -c "from models.board_mamba import HAS_MAMBA, BoardMamba; print(BoardMamba().model_name); assert HAS_MAMBA, 'Mamba backend unavailable'"
```

本次 Windows 驗收沒有重新安裝或驗證 CUDA Mamba。Fallback 結果不得當作 BoardMamba 成績。

## 資料、訓練與搜尋

以下步驟會下載資料或執行較長工作；先檢查 [configs](configs/) 中的樣本數、訓練輪數、模擬次數和 checkpoint 路徑。

```bash
bash scripts/download_lichess_sample.sh
bash scripts/prepare_policy_dataset.sh
export STOCKFISH_PATH=/path/to/stockfish
bash scripts/generate_stockfish_values.sh
bash scripts/train_board_mamba.sh
bash scripts/train_cnn_baseline.sh
```

資料按整局切分，避免同一棋局的位置分散到不同資料集。MCTS、head-to-head 和 self-play 需要相應的 joint policy/value checkpoint；此 repo 不附舊模型權重、棋譜、Stockfish 執行檔或大型產物。

## 歷史結果

下表取自 [joint pilot CSV](evidence/joint_pilot_summary.csv)，兩個模型均為表內的 epoch 1 設定。它是過去的 pilot 紀錄，未在本次整理時重跑。

| 模型 | Validation top-1 ↑ | Value MAE ↓ |
|---|---:|---:|
| BoardMamba joint | 0.1491 | 0.3808 |
| CNN joint | 0.1807 | 0.3347 |

其他保留結果：[policy/value pilot](evidence/pilot_summary.csv)、[value pilot](evidence/value_pilot_summary.csv)。小樣本、短訓練和特定對手不足以推估 Elo，也不能主張搜尋或 self-play 已普遍改善棋力。下一步是固定完整環境與資料切分，補上可下載 checkpoint 和獨立重現。

## 來源與使用範圍

- [Mamba](https://github.com/state-spaces/mamba)：外部模型元件。
- [python-chess](https://github.com/niklasf/python-chess)：規則、PGN 與 UCI 介面，依其 GPL 授權。
- [Lichess open database](https://database.lichess.org/)：棋譜來源。
- [Stockfish](https://github.com/official-stockfish/Stockfish)：另行安裝的引擎，依其 GPL 授權。

這是既有實驗專案的公開整理版，保留原程式與選定 CSV；機器專用啟動路徑改為可攜路徑。第三方套件、引擎和資料未隨 repo 重散布。原創部分未授予額外使用授權；詳見 [NOTICE](NOTICE.md)。
