# 動物分類器による人間の顔の特徴に対する分類技術レポート
## A Computational Classification of Human Facial Traits

**日付**: 2025年7月12日
**プロジェクト**: 動物顔特徴分類システム
**最終実行日**: 2025年7月12日 03:07:11

---

## 1. データ収集方法と前処理方法

### 1.1 データ収集方法

#### 1.1.1 動物画像データセット
- **データソース**: Kaggle Animal Faces Dataset (`andrewmvd/animal-faces`)
- **収集方法**: Kaggle APIを使用した自動ダウンロード
- **データ構造**: 
  - Cat（猫）: 5,153枚の動物の顔画像
  - Dog（犬）: 4,739枚の動物の顔画像  
  - Wild（野生動物）: 4,738枚の野生動物の顔画像
- **合計**: 14,630枚の動物画像（訓練用）
- **実装**: `scripts/download_kaggle_dataset.py`で自動化

#### 1.1.2 人間-動物ハイブリッド画像
- **生成方法**: AI画像生成サービス（Pollinations.ai API, Imagine4, ChatGPT）を使用
- **生成数**: 30枚（各動物タイプ10枚ずつ）
- **ファイル命名規則**: `{animal_name}_human_{number}.jpg`
  - 例: `cat_human_01.jpg`, `dog_human_05.jpg`, `wild_human_03.jpg`
- **生成プロンプト例**:
  ```
  Cat-like humans:
  - "Color image of a human face with a slight cat-like appearance."
  - "Color image of a human face with a slight cat-like appearance like anime."
  
  Dog-like humans:
  - "Color image of a human face with a slight dog-like appearance."
  - "Color image of a human face with a slight dog-like appearance like anime."
  
  Wild animal-like humans:
  - "Color image of a human face with a slight tiger-like appearance."
  - "Color image of a human face with a slight tiger-like appearance like anime."
  ```
- **実装**: `scripts/generate_human_animal_images.py`で自動生成, ブラウザで手動生成
- **特徴**: 現実とアニメ風での動物的特徴を持つ人間の顔画像

### 1.2 前処理方法

#### 1.2.1 画像前処理パイプライン
1. **画像読み込み**: OpenCV (`cv2.imread()`)
2. **色空間変換**: BGR → RGB → グレースケール
3. **サイズ統一**: 128×128ピクセルにリサイズ（`cv2.resize()`）
4. **切り取り処理**: **実施せず** - 全画像を128×128に統一リサイズのみ
5. **フラット化**: 16,384次元ベクトルに変換
6. **正規化**: 0-1範囲への正規化 (`pixel_value / 255.0`)

#### 1.2.2 実際の前処理パラメータ
- **入力画像サイズ**: 128×128ピクセル
- **特徴ベクトル長**: 16,384次元
- **正規化範囲**: [0, 1]
- **データ分割**: 訓練80% (11,704枚) / テスト20% (2,926枚)
- **切り取り**: なし（全画像を統一サイズにリサイズのみ）

---

## 2. PCA（主成分分析）の実装

### 2.1 使用ライブラリと実装詳細

#### 2.1.1 ライブラリ
- **scikit-learn**: `sklearn.decomposition.PCA`
- **機能**: 主成分分析による次元削減

#### 2.1.2 実装コード例
```python
from sklearn.decomposition import PCA

# PCAオブジェクトの作成
pca = PCA(n_components=110)
X_pca = pca.fit_transform(X)

# 寄与率の計算
explained_variance_ratio = pca.explained_variance_ratio_.sum()
```

### 2.2 ハイパーパラメータとその影響

#### 2.2.1 主成分数の決定
- **選択した成分数**: 110成分
- **決定根拠**: 80%以上の情報保持を目標
- **実際の寄与率**: **80.89%**
- **圧縮率**: 16,384 → 110 (約149倍の圧縮)

#### 2.2.2 成分数の影響分析
- **少なすぎる場合**: 情報損失が大きく、分類精度が低下
- **多すぎる場合**: ノイズを含み、過学習のリスク
- **最適値**: 80-90%の寄与率を保持する成分数

### 2.3 PCA結果の可視化

#### 2.3.1 固有顔（Eigenfaces）
- **生成数**: 上位6成分の可視化
- **保存場所**: `results/20250712_000423/eigenfaces.png`
- **解釈**: 各成分は顔の異なる特徴パターンを表現
- **特徴**: 各固有顔は異なる顔の特徴（輪郭、目、鼻、口など）を強調

#### 2.3.2 寄与率プロット
- **累積寄与率**: 成分数に対する累積寄与率の推移
- **個別寄与率**: 上位20成分の個別寄与率
- **保存場所**: `results/20250712_000423/pca_analysis.png`

---

## 3. SVM（サポートベクターマシン）の実装

### 3.1 使用ライブラリ

#### 3.1.1 ライブラリ
- **scikit-learn**: `sklearn.svm.SVC`
- **グリッドサーチ**: `sklearn.model_selection.GridSearchCV`
- **前処理**: `sklearn.preprocessing.StandardScaler`
- **評価**: `sklearn.metrics`（各種評価指標）

#### 3.1.2 実装コード例
```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# 標準化（重要：SVMは特徴量のスケールに敏感）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)

# SVM実装（確率推定有効）
svm = SVC(probability=True)
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
```

### 3.2 ハイパーパラメータとその影響

#### 3.2.1 最適化されたパラメータ
```python
# 実際の最適パラメータ
best_params = {
    'C': 10,           # 正則化パラメータ
    'gamma': 0.01,     # RBFカーネルパラメータ
    'kernel': 'rbf',   # 放射基底関数カーネル
    'probability': True # 確率推定有効
}
```

#### 3.2.2 グリッドサーチ範囲
```python
param_grid = {
    'C': [0.1, 1, 10, 100],  # 正則化パラメータ
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],  # RBFカーネルパラメータ
    'kernel': ['rbf', 'poly', 'sigmoid']  # カーネル関数
}
```

#### 3.2.3 パラメータの影響
- **C値**: 正則化強度
  - 高い値: 複雑な決定境界、過学習のリスク
  - 低い値: 単純な決定境界、汎化性能向上
- **gamma値**: RBFカーネルの影響範囲
  - 高い値: 局所的な決定境界、過学習のリスク
  - 低い値: 滑らかな決定境界、汎化性能向上
- **kernel種類**: 
  - RBF: 非線形分離に適している
  - Poly: 多項式による非線形分離
  - Sigmoid: ニューラルネットワーク類似

---

## 4. 実験2で用いる動物ライクな人の顔画像

### 4.1 収集とラベル付け方法

#### 4.1.1 AI生成による画像収集
- **生成ツール**: Pollinations.ai API（無料）, Imagine4, ChatGPT
- **生成方法**: テキストプロンプトによる画像生成
- **成功率**: 低い（何回も生成して適切なものを選択）
- **選択基準**: 現実的、アニメ風など幅広い世界観で、動物過ぎず人間過ぎない、適度な動物的特徴を持つもの

#### 4.1.2 ラベル付け方法
- **アプローチ**: 生成時に基づくファイル名によるラベル
- **命名規則**: `{animal_name}_human_{number}.jpg`
- **3クラス分類**: cat-like, dog-like, wild-like
- **クラスマッピング**: `{'cat': 0, 'dog': 1, 'wild': 2}`

### 4.2 具体的な生成例と分布

#### 4.2.1 生成画像の分布
- **Cat-like**: 10枚（cat_human_01.jpg ～ cat_human_10.jpg）
- **Dog-like**: 10枚（dog_human_01.jpg ～ dog_human_10.jpg）
- **Wild-like**: 10枚（wild_human_01.jpg ～ wild_human_10.jpg）

#### 4.2.2 生成プロンプトの詳細
```
Cat-like humans:
- "Color image of a human face with a slight cat-like appearance."
- "Color image of a human face with a slight cat-like appearance like anime."

Dog-like humans:
- "Color image of a human face with a slight dog-like appearance."
- "Color image of a human face with a slight dog-like appearance like anime."

Wild animal-like humans:
- "Color image of a human face with a slight tiger-like appearance."
- "Color image of a human face with a slight tiger-like appearance like anime."
```

---

## 5. 実験結果

### 5.1 モデル評価結果

#### 5.1.1 実際のデータセット規模
```python
# 確定データ数
dataset_summary = {
    'total_animal_images': 14630,
    'total_human_images': 30,
    'train_size': 11704,
    'test_size': 2926,
    'image_size': (128, 128),
    'feature_dim_original': 16384,
    'feature_dim_pca': 110
}
```

#### 5.1.2 実際の前処理パラメータ
- **PCA成分数**: 110成分（16,384次元から110次元へ削減）
- **画像サイズ**: 128×128ピクセル
- **正規化**: 0-1範囲への正規化
- **グレースケール変換**: 実装済み
- **寄与率**: **80.89%**（110成分で全体の約81%の情報を保持）

### 5.2 PCAの可視化結果（**完了**）

#### 5.2.1 主成分分析結果
```python
# 実際の結果
pca_results = {
    'original_dimensions': 16384,
    'reduced_dimensions': 110,
    'explained_variance_ratio': 0.8089,  # 80.89%
    'compression_ratio': 149.0  # 約149倍の圧縮
}
```

#### 5.2.2 固有顔の可視化（**完了**）
- **保存場所**: `results/20250712_000423/eigenfaces.png`
- **内容**: 上位6つの主成分を画像として表示
- **解釈**: 各固有顔は異なる顔の特徴（輪郭、目、鼻、口など）を強調
- **特徴**: PCAの固有ベクトルをグレースケール画像として可視化

#### 5.2.3 2D散布図（**完了**）
- **保存場所**: `results/20250712_000423/pca_2d_visualization.png`
- **内容**: 第1・第2主成分での各クラスの分布
- **色分け**: 3クラス（cat, dog, wild）ごとに異なる色
- **観察**: クラス間の分離性と重複領域の可視化

### 5.3 SVM分類結果（**可視化以外完了**）

#### 5.3.1 最適化されたハイパーパラメータ
```python
# 実際の最適パラメータ
best_params = {
    'C': 10,           # 正則化パラメータ
    'gamma': 0.01,     # RBFカーネルパラメータ
    'kernel': 'rbf',   # 放射基底関数カーネル
    'probability': True # 確率推定有効
}

# 交差検証スコア: 86.61%
# テスト精度: 87.29%
```

#### 5.3.2 評価指標（**完了**）
- **混同行列**: `results/20250712_000423/confusion_matrix.png`
- **全体精度**: **87.29%**
- **交差検証精度**: **86.61%**

### 5.4 実験1: 各動物クラスごとの精度（**完了**）

#### 5.4.1 実際の分類結果
```python
# 各クラスの詳細性能（動物分類）
classification_results = {
    'cat': {
        'precision': 0.88,
        'recall': 0.88,
        'f1_score': 0.88,
        'support': 1030
    },
    'dog': {
        'precision': 0.86,
        'recall': 0.90,
        'f1_score': 0.88,
        'support': 948
    },
    'wild': {
        'precision': 0.88,
        'recall': 0.84,
        'f1_score': 0.86,
        'support': 948
    }
}
```

#### 5.4.2 クラス別性能分析
1. **Cat（猫）**: 安定した性能
   - 精密度: 88%、再現率: 88%、F1スコア: 88%
   - 大量のデータ（5,153枚）による安定した学習

2. **Dog（犬）**: 高い再現率
   - 精密度: 86%、再現率: 90%、F1スコア: 88%
   - 高い再現率（見逃しが少ない）

3. **Wild（野生動物）**: バランスの取れた性能
   - 精密度: 88%、再現率: 84%、F1スコア: 86%
   - 多様な動物種を含むにも関わらず良好

### 5.5 実験2: 人間-動物分類結果（**完了**）

#### 5.5.1 AI生成画像の分類結果
```python
# 人間-動物ハイブリッド画像の分類結果
human_animal_classification = {
    'total_images': 30,
    'accuracy': 0.3667,  # 36.67%
    'correct_predictions': 11,
    'incorrect_predictions': 19,
    'dog_like_predictions': 23,  # 76.7%
    'cat_like_predictions': 4,   # 13.3%
    'wild_like_predictions': 3,  # 10.0%
}
```

#### 5.5.2 詳細な分類結果分析

**Cat-like Human画像の分類結果**:
- 生成画像10枚中、実際の分類結果:
  - **正解**: 0枚 (0%) - 全て誤分類
  - Dog-like: 10枚 (100%)
  - Cat-like: 0枚 (0%)
  - Wild-like: 0枚 (0%)

**Dog-like Human画像の分類結果**:
- 生成画像10枚中、実際の分類結果:
  - **正解**: 9枚 (90%) - 最も高い精度
  - Dog-like: 9枚 (90%)
  - Wild-like: 1枚 (10%)
  - Cat-like: 0枚 (0%)

**Wild-like Human画像の分類結果**:
- 生成画像10枚中、実際の分類結果:
  - **正解**: 2枚 (20%) - 低い精度
  - Dog-like: 4枚 (40%)
  - Cat-like: 4枚 (40%)
  - Wild-like: 2枚 (20%)

#### 5.5.3 分類の信頼度分析
- **高信頼度分類** (>0.9): 13枚 (43.3%)
- **中信頼度分類** (0.5-0.9): 15枚 (50.0%)
- **低信頼度分類** (<0.5): 2枚 (6.7%)

#### 5.5.4 分類傾向の分析
- **Dog-like偏重**: 76.7%の画像がdog-likeと分類
- **Cat-like検出困難**: 生成時のプロンプトに基づいたラベル付けにおけるcat-likeの画像は全て誤分類
- **Wild-like識別困難**: 生成時のプロンプトに基づいたラベル付けにおけるwild-likeの画像の80%が誤分類

#### 5.5.5 個別画像の分類結果詳細

**Cat-like画像（全て誤分類）**:
- cat_human_01.jpg: dog-like (信頼度: 1.000)
- cat_human_02.jpg: dog-like (信頼度: 0.805)
- cat_human_06.jpg: dog-like (信頼度: 0.491) - 最低信頼度
- cat_human_10.jpg: dog-like (信頼度: 0.989)

**Dog-like画像（90%正解）**:
- dog_human_01.jpg: wild-like (信頼度: 0.546) - 唯一の誤分類
- dog_human_02.jpg: dog-like (信頼度: 0.655) - 最低信頼度
- dog_human_10.jpg: dog-like (信頼度: 0.994) - 最高信頼度

**Wild-like画像（20%正解）**:
- wild_human_01.jpg: wild-like (信頼度: 0.925) ✓
- wild_human_08.jpg: wild-like (信頼度: 0.980) ✓
- wild_human_04.jpg: cat-like (信頼度: 0.289) - 最低信頼度
- wild_human_05.jpg: cat-like (信頼度: 0.970) - 高信頼度誤分類

### 5.6 実験3: 用いたモデルでの総合評価（**完了**）

#### 5.6.1 実装アプローチの効果
- **動物分類**: 87.29%の高精度を達成
- **人間評価**: 36.67%の精度（課題あり）
- **データ不均衡問題**: 解決済み（分離アプローチ）

#### 5.6.2 モデルの特性分析
```python
# 総合的なモデル性能
model_performance = {
    'animal_classification_accuracy': 87.29,  # 動物分類は高精度
    'human_classification_accuracy': 36.67,  # 人間分類は課題あり
    'dog_like_bias': 76.7,  # Dog-like特徴への強い偏り
    'cat_like_detection': 0.0,  # Cat-like特徴の検出困難
    'wild_like_detection': 20.0,  # Wild-like特徴の限定的検出
}
```

#### 5.6.3 分類偏重の原因分析
1. **訓練データの影響**: 動物の顔で訓練されたモデルが人間の顔に適用
2. **特徴空間の違い**: 動物と人間の顔の特徴空間が異なる
3. **Dog-like特徴の汎用性**: 人間の表情筋がdog-like特徴と類似
4. **Cat/Wild特徴の特異性**: より特異的な特徴で検出困難

---

## 6. 総合評価と考察

### 6.1 技術的成果 ✅

#### 6.1.1 高精度分類システム
- **動物分類精度**: **87.29%**（3クラス分類）
- **交差検証精度**: **86.61%**（安定性確認）
- **大規模データ処理**: 14,630枚の画像処理成功

#### 6.1.2 効果的な次元削減
- **PCA圧縮率**: 149倍（16,384 → 110次元）
- **情報保持率**: **80.89%**
- **計算効率**: 大幅な処理時間短縮

### 6.2 実験設計の成果と課題 ⚠️

#### 6.2.1 成功した点
- **データ不均衡問題の解決**: 3クラス動物分類 + 人間評価分離
- **安定した動物分類**: 87.29%の高精度
- **AI生成画像の活用**: 30枚の評価用画像生成成功

#### 6.2.2 発見された課題
- **人間分類の低精度**: 36.67%（期待値33.3%をわずかに上回る）
- **Dog-like偏重**: 76.7%の画像がdog-likeと分類される偏り
- **Cat-like検出不能**: 生成時のプロンプトに基づいたラベル付けにおけるcat-likeの画像は全て誤分類

### 6.3 発見された知見 💡

#### 6.3.1 ドメイン適応の困難性
- **特徴空間の違い**: 動物の顔と人間の顔の特徴空間が大きく異なる
- **転移学習の限界**: 動物で訓練されたモデルの人間への適用困難
- **Dog-like特徴の汎用性**: 人間の表情筋がdog-like特徴と最も類似

#### 6.3.2 AI生成画像の特性
- **生成品質の影響**: AI生成画像の特徴が実際の人間の顔と異なる可能性
- **プロンプトの影響**: 生成プロンプトが意図した特徴を正確に反映しない
- **評価の困難性**: 「動物的特徴」の主観性と定量化の困難

### 6.4 技術的洞察 🔍

#### 6.4.1 PCAの有効性
- **大幅な次元削減**: 99.3%の次元削減でも高精度維持（動物分類）
- **ノイズ除去効果**: 主要な特徴のみを抽出
- **計算効率**: 訓練・推論時間の大幅短縮

#### 6.4.2 SVMの適用性
- **動物分類での成功**: 非線形分離による高精度分類
- **人間分類での課題**: 特徴空間の違いによる性能低下
- **確率推定の有用性**: 分類の信頼度評価が可能

### 6.5 今後の改善方向 🚀

#### 6.5.1 技術的改善
- **ドメイン適応**: 動物と人間の特徴空間を橋渡しする手法
- **ファインチューニング**: 人間の顔データでの追加学習
- **マルチモーダル学習**: 複数の特徴抽出手法の組み合わせ

#### 6.5.2 データ改善
- **実写人間画像**: AI生成ではなく実際の人間の顔画像
- **専門家ラベリング**: 動物的特徴の客観的評価
- **データ拡張**: より多様な人間の顔画像の収集

---

## 7. 技術的成果（確定）

### 7.1 システム実装 ✅
- **完全実装**: 3クラス動物分類システム（87.29%精度）
- **大規模データ処理**: 14,630枚の画像処理
- **最適化**: グリッドサーチによるハイパーパラメータ最適化
- **可視化**: 固有顔、混同行列、PCA散布図の生成

### 7.2 実験規模（確定） ✅
- **データ数**: 14,630枚（cat: 5,153, dog: 4,739, wild: 4,738）
- **人間画像**: 30枚（AI生成）
- **特徴量**: 16,384次元 → 110次元（80.89%情報保持）
- **分類器**: SVM（C=10, gamma=0.01, RBF kernel, probability=True）
- **評価**: 5-fold交差検証（86.61%）、テスト精度（87.29%）

### 7.3 学術的貢献 ✅
1. **技術的**: PCA+SVMによる高精度動物顔分類システム
2. **方法論的**: 大規模データセットでの次元削減効果の実証
3. **課題発見**: ドメイン適応の困難性と偏重問題の発見
4. **問題解決**: データ不均衡問題の効果的解決方法の提示

### 7.4 実用的応用と課題 ⚠️
- **成功事例**: 動物分類システム（87.29%精度）
- **課題**: 人間の動物的特徴分類（36.67%精度）
- **改善の必要性**: ドメイン適応とデータ品質の向上
- **将来性**: 技術改善により実用化可能

---

**最終更新**: 2025年7月12日  
**実験ステータス**: 完了  
**動物分類精度**: 87.29%  
**人間分類精度**: 36.67%  
**データセット**: 14,630枚（3クラス動物分類）  
**人間画像評価**: 30枚（100%処理成功、36.67%分類精度）  
**技術スタック**: Python, scikit-learn, OpenCV, Kaggle API, AI画像生成サービス API  
**主要発見**: Dog-like特徴への偏重（76.7%）、Cat-like特徴検出困難（0%精度） 