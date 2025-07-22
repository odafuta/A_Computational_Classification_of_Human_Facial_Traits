# 動物顔特徴分類システム 技術レポート (20250719_161924 更新版)
## A Computational Classification of Human Facial Traits

**日付**: 2025年7月19日  
**対象スクリプト**: `main_simple.py`  
**モデル ID**: `20250719_161924`

---

## 0. 概要
本プロジェクトは，猫・犬・トラの 3 クラス動物顔画像で学習した分類器を，人間の顔画像にも適用して「どの動物に近いか」を推定する試みである。最新モデルは交差検証 **83.1 %**，検証 **81.8 %**，テスト **87.7 %** の精度を達成した。

---

## 1. データセットと前処理
- **動物画像数**: 1,350 枚（訓練: 70% 944枚 / 検証: 15% 203枚 / テスト: 15% 203枚）  
- **人間画像数**: 30 枚（AI 生成，ファイル名でラベル付け）  
- **前処理**: RGB → グレースケール → 128×128 リサイズ → 0–1 正規化 → 平坦化 (16,384 次元)

---

## 2. モデル構成
```
StandardScaler → PCA(n_components=110, explained_variance=0.823) → SVC(kernel='rbf', C=10, γ='scale')
```
- PCA で 149 倍の次元圧縮（16,384 → 110）
- ハイパーパラメータは GridSearchCV (5-fold) で最適化

### 2.1 実装フロー（擬似コード抜粋）
```python
# main_simple.py から主要部分を簡略化

def main():
    setup_plotting()                        # 0) 描画設定
    args = parse_arguments()                # 1) CLI 引数を取得
    timestamp = now()
    models_root, results_dir = create_directories(timestamp)

    # 2) ログを tee でファイルにも保存
    redirect_stdout_to(results_dir / 'experiment_log.txt')

    # 3) データロード
    X_animal, y_animal, _   = load_animal_data()
    X_human,  y_human, fns = load_human_data()

    # 4) モデルロード or 学習
    if args.use_existing or args.model_dir:
        pipeline, metadata = load_model_and_metadata(...)
    else:
        pipeline, metrics  = train_model_with_validation(
            X_animal, y_animal, results_dir)
        save_model_and_metadata(pipeline, metrics, timestamp, models_root)

    # 5) 可視化
    visualize_pca_analysis(X_animal, y_animal, pipeline.named_steps['pca'], results_dir)
    if not args.skip_boundary:
        visualize_svm_decision_boundary(X_animal, y_animal, pipeline, results_dir)

    # 6) 評価
    evaluate_animal_test_set(pipeline, X_animal, y_animal, results_dir)
    evaluate_human_faces(pipeline, X_human, y_human, fns, results_dir)

    print('=== Experiment Complete ===')

# パイプライン生成関数

def create_model_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('pca',    PCA(n_components=110)),
        ('svc',    SVC(kernel='rbf'))
    ])
```
上記の擬似コードにより **main_simple.py** の全体の制御フロー（データ→学習→評価→保存・可視化）が把握できる。

---

## 3. 評価指標（`models/20250719_161924/metadata.json` より）
| 指標 | 値 |
|------|------|
| 交差検証平均精度 (`cv_score`) | **0.8305** |
| 交差検証標準偏差 (`cv_std`) | 0.0195 |
| 検証精度 (`val_score`) | **0.8177** |
| テスト精度 (`test_score`) | **0.8768** |
| PCA 累積寄与率 | 0.8230 |

### 3.1 動物テストセット分類レポート
| クラス | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| cat   | 0.84 | 0.91 | 0.87 | 67 |
| dog   | 0.90 | 0.81 | 0.85 | 68 |
| tiger | 0.90 | 0.91 | 0.91 | 68 |
| **全体精度** | \- | \- | **0.88** | 203 |

### 3.2 人間画像分類レポート（30 枚）
| クラス | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| cat   | 0.12 | 0.10 | 0.11 | 10 |
| dog   | 0.35 | 0.70 | 0.47 | 10 |
| tiger | 1.00 | 0.20 | 0.33 | 10 |
| **全体精度** | \- | \- | **0.33** | 30 |

---

## 4. 結果ファイル一覧（`results/20250719_161924/`）
| # | パス | ファイル | 役割・解釈方法 |
|---|---|---|---|
| 1 | `results/20250719_161924/` | `experiment_log.txt` | 実行時の標準出力/エラーを全て保存したログ。ハイパーパラメータ探索やスコア推移を再現可能。|
| 2 | 〃 | `pca_analysis.png` | 左: 累積寄与率，右: 各主成分の寄与率 (上位20)。80% を超える位置で 110 成分を採用した根拠を確認可能。|
| 3 | 〃 | `eigenfaces.png` | 上位 6 主成分を画像化したもの。顔画像として可視化することで PCA が抽出した特徴を直感的に理解できる。|
| 4 | 〃 | `pca_2d_visualization.png` | 第1・第2主成分で散布図化し，各クラス (cat/dog/tiger) を色分け。クラス間分離の程度を視覚的に評価。|
| 5 | 〃 | `svm_decision_boundary.png` | PCA 2 次元空間での SVM 決定境界。データ点と合わせて可視化し，境界形状と誤分類領域を確認。|
| 6 | 〃 | `animal_confusion_matrix.png` | 動物テストセットの混同行列。行 (実ラベル) と列 (予測) の対応を見て誤分類パターンを把握。|
| 7 | 〃 | `animal_classification_summary.png` | テストセットに対する予測クラス分布バーグラフ。クラス不均衡や偏りを定性的に確認。|
| 8 | 〃 | `animal_classification_report.txt` | Precision / Recall / F1 / Support を含むテキストレポート（sklearn 出力）。数値的な性能比較に利用。|
| 9 | 〃 | `human_confusion_matrix.png` | 人間画像 30 枚の分類混同行列。モデルがどの動物クラスを選びやすいかを可視化。|
|10 | 〃 | `human_classification_summary.png` | 人間画像に対する予測クラス分布バーグラフ。dog 偏重などの傾向を把握。|

---

## 5. 考察
- **テスト精度向上**: 前回モデルよりテスト精度が +0.9 pt 改善。検証 > テスト の逆転はデータ分布差解消を示唆。
- **クラス間誤分類**: 混同行列より dog ↔ tiger 間で誤分類が残る。さらなる特徴量追加が必要。
- **人間画像評価**: dog クラスへの偏重が依然として存在。ドメイン適応や追加学習が課題。

---

## 6. 今後の改善案
1. **ハイパーパラメータ探索の拡張**: `C` と `γ` の広域探索＋Bayesian Optimization を試行。  
2. **データ拡張**: 回転・色調変化などを加えデータ多様性を向上。  
3. **ドメイン適応**: 人間顔データでファインチューニングし，動物→人間のギャップを縮小。  
4. **他モデル比較**: RandomForest，CNN などと性能・計算コストを比較。  

---

**最終更新**: 2025年7月19日  
**動物分類精度**: 88%  
**人間分類精度**: 33%  
**データセット**: 1350枚（3クラス動物分類）  
**人間画像評価**: 30枚（37%分類精度）  
**技術スタック**: Python, scikit-learn, OpenCV, Kaggle API, AI画像生成サービス API  
**主要発見**: Dog-like特徴への偏重、Cat-like特徴検出困難