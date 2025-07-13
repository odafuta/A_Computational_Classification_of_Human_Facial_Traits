# Assignment 2 Improvement Plan
## A Computational Classification of Human Facial Traits - Modifications and Additions for Completion

---

## 1. Current Issues and Challenges

### 1.1 Environment and Dependency Issues
- **Missing dependency libraries**: scikit-learn, PIL (Pillow), matplotlib, etc. are not installed
- **Missing requirements.txt**: Required libraries and versions are not specified
- **Unorganized Python environment**: Virtual environment setup is unclear

### 1.2 Dataset Issues
- **Insufficient data**: Only about 16 images per animal category (comments mention 450 images)
- **Missing human face data**: The original purpose is to "classify human faces into animals" but currently only animal faces exist
- **Data imbalance**: Possible inconsistency in data count between categories

### 1.3 Code Issues
- **Insufficient error handling**: No handling for missing files
- **Global variable dependency**: Undefined `y` variable referenced in `figPCA` function
- **Incomplete visualization features**: PCA visualization and SVM boundary display may not work
- **Confusing training/inference mode switching**: Multiple code sections need commenting/uncommenting
- **Missing model saving functionality**: No save/load functionality for trained models

### 1.4 Deviation from Assignment Requirements
- **Need for human face image data**: Currently only animal faces
- **Application development**: GUI/Web app for bonus points not implemented
- **4-page report**: No academic paper format template

---

## 2. Required Modifications and Additions

### 2.1 Environment Setup 【High Priority】

#### A. Create requirements.txt
```
numpy>=1.21.0
scikit-learn>=1.0.0
Pillow>=8.0.0
matplotlib>=3.5.0
opencv-python>=4.5.0
streamlit>=1.0.0  # For web app (optional)
```

#### B. Virtual Environment Setup Procedure
```bash
python -m venv facial_classification_env
facial_classification_env\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2.2 Dataset Expansion 【High Priority】

#### A. Human Face Dataset Collection
- **Recommended datasets**:
  - CelebA dataset (celebrity faces)
  - LFW (Labeled Faces in the Wild)
  - Custom dataset (authorized person photos)

#### B. Data Directory Structure Change
```
af_data/
├── human_faces/          # Human face data (main)
│   ├── cat_like/         # Cat-like human faces
│   ├── dog_like/         # Dog-like human faces
│   ├── fox_like/         # Fox-like human faces
│   └── tiger_like/       # Tiger-like human faces
└── reference_animals/    # Reference animal faces
    ├── cat/
    ├── dog/
    ├── fox/
    └── tiger/
```

#### C. Data Augmentation
- Target minimum 100+ images per category
- Image enhancement through rotation, flipping, brightness adjustment (if time permits)

### 2.3 Code Structure Improvement 【High Priority】

#### A. Separate Training and Inference Functions

**1. Create training function**
```python
def train_model():
    """Training mode - trains PCA and SVM models"""
    print("Starting training mode...")
    x, y = imgLoad()
    if x is None or y is None:
        print("Failed to load data")
        return
    
    # Train model
    acc, pca, svm = xSVM(x, y)
    
    # Optional visualizations
    figPCA(x, y, eigen=True, mean=True, com=100)
    figSVM(x, y)
    
    print(f"Training completed. Accuracy: {acc}")
    return pca, svm
```

**2. Create inference function**
```python
def inference_mode():
    """Inference mode - loads pre-trained models and evaluates"""
    print("Starting inference mode...")
    try:
        pca = joblib.load('./pca_110.pkl')
        svm = joblib.load('./svm_c1.5_com110.pkl')
        imgEva()
    except FileNotFoundError as e:
        print(f"Model files not found: {e}")
        print("Please run training mode first")
```

**3. Clean main function**
```python
if __name__ == "__main__":
    # Switch between modes by commenting/uncommenting ONE line
    
    # Training mode - uncomment this line
    # train_model()
    
    # Inference mode - uncomment this line
    inference_mode()
```

#### B. Function Parameter Fixes

**1. Fix figPCA function**
```python
def figPCA(x, y=None, eigen=False, mean=False, com=100):
    pca = PCA(n_components=com)
    x_pca = pca.fit_transform(x)
    print("retain rate:", sum(pca.explained_variance_ratio_))
    
    if eigen:
        eigen1 = pca.components_[0].reshape((128, 128))
        eigen2 = pca.components_[1].reshape((128, 128))
        plt.subplot(1, 2, 1)
        plt.imshow(eigen1, cmap='gray')
        plt.title('eigen 1')
        plt.subplot(1, 2, 2)
        plt.imshow(eigen2, cmap='gray')
        plt.title('eigen 2')
        plt.show()
    
    if mean and y is not None:
        labels = ['cat', 'dog', 'fox', 'tiger']
        colors = ['red', 'green', 'blue', 'orange']
        for i in range(4):
            plt.scatter(
                x_pca[y == i, 0],
                x_pca[y == i, 1],
                label=labels[i],
                color=colors[i],
                alpha=0.6,
                edgecolor='k'
            )
        plt.legend()
        plt.show()
    
    return x_pca, pca
```

**2. Fix figSVM function**
```python
def figSVM(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    x_train_2d, pca_vis = figPCA(x_train, y_train, False, False, 2)
    svm_vis = SVC(kernel='rbf', C=10, gamma='scale')
    svm_vis.fit(x_train_2d, y_train)
    
    # Create visualization
    h = 1.0
    x_min, x_max = x_train_2d[:, 0].min() - 1, x_train_2d[:, 0].max() + 1
    y_min, y_max = x_train_2d[:, 1].min() - 1, x_train_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    Z = svm_vis.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    labels = ['cat', 'dog', 'fox', 'tiger']
    colors = ['red', 'green', 'blue', 'orange']
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    for i in range(4):
        plt.scatter(
            x_train_2d[y_train == i, 0],
            x_train_2d[y_train == i, 1],
            c=colors[i],
            label=labels[i],
            edgecolor='k',
            alpha=0.8
        )
    plt.legend()
    plt.title('SVM Decision Boundary (2D PCA)')
    plt.show()
```

**3. Fix accTest function**
```python
def accTest(x, y, lowN=120, highN=121, stepN=1, lowC=2, highC=3, stepC=1):
    name = str(lowN) + '_' + str(highN) + '_' + str(stepN)
    txt = []
    for i in range(lowN, highN, stepN):
        for j in range(int(lowC / stepC), int(highC / stepC), 1):
            a, _, _ = xSVM(x, y, i, j * stepC)
            res = str(a) + ' > components = ' + str(i) + ', C = ' + str(j * stepC) + '\n'
            print(res)
            txt.append(res)
    with open('./res' + str(name) + '.txt', 'w') as file:
        file.writelines(txt)
```

### 2.4 Basic Error Handling 【Medium Priority】

#### A. Add basic error handling
```python
def imgLoad():
    try:
        images = []
        labels = []
        for id, animal in enumerate(['cat', 'dog', 'fox', 'tiger']):
            class_path = './af_data/' + str(animal)
            if not os.path.exists(class_path):
                print(f"Warning: Directory {class_path} not found")
                continue
            
            files = os.listdir(class_path)
            if len(files) == 0:
                print(f"Warning: No files in {class_path}")
                continue
                
            for i in files:
                try:
                    img = Image.open(class_path + '/' + i).convert('L')
                    img = img.resize((128, 128))
                    imgArr = np.array(img).flatten()
                    images.append(imgArr)
                    labels.append(id)
                except Exception as e:
                    print(f"Error loading {class_path}/{i}: {e}")
                    continue
        
        if len(images) == 0:
            print("No images loaded successfully")
            return None, None
            
        return np.array(images), np.array(labels)
    except Exception as e:
        print(f"Error in imgLoad: {e}")
        return None, None
```

### 2.5 Documentation 【Medium Priority】

#### A. README.md Update
- Project overview
- Installation instructions
- Usage instructions (training vs inference)
- Dataset information

#### B. Code Comments
- Add clear comments explaining each function
- Document the training/inference mode switching

---

## 3. Implementation Priority

### Phase 1: Foundation Setup (Week 1)
1. Environment construction (requirements.txt, virtual environment)
2. Human face dataset collection and organization
3. Code structure improvement (separate training/inference functions)
4. Basic error handling

### Phase 2: Testing and Refinement (Week 2)
1. Test with collected human face data
2. Fix any remaining bugs
3. Optimize parameters if needed
4. Create documentation

### Phase 3: Completion and Optimization (Week 3)
1. Web application development (for bonus)
2. Report creation
3. Final testing and debugging
4. Presentation preparation

---

## 4. Recommended Team Assignment

- **Project Lead**: Overall progress management, code integration
- **Data Collector**: Human face dataset collection and preprocessing
- **Data Analyst**: PCA/SVM optimization, testing
- **App Developer**: Web application development (for bonus)
- **Report Writer**: Academic paper format report creation
- **Presenter**: Presentation preparation and delivery

---

## 5. Success Metrics

### Assignment Requirements
- 4-page academic paper completion
- Team contribution documentation
- Code submission with clear structure
- On-time submission
- Presentation delivery

### Bonus (if time permits)
- Web application functionality
- User-friendly UI
- Real-time classification functionality

---

# Assignment 2 改善計画書
## A Computational Classification of Human Facial Traits - 完成に向けた修正・追加事項

---

## 1. 現在の問題点と課題

### 1.1 環境・依存関係の問題
- **依存ライブラリ未インストール**: scikit-learn、PIL (Pillow)、matplotlib等が未インストール
- **requirements.txtが存在しない**: 必要なライブラリとバージョンが明記されていない
- **Python環境の未整備**: 仮想環境の設定が不明確

### 1.2 データセットの問題
- **データ量不足**: 各動物カテゴリ16枚程度（コメントでは450枚と記載）
- **人間の顔データなし**: 課題の本来の目的は「人間の顔を動物に分類」だが、現在は動物の顔のみ
- **データの不均衡**: カテゴリ間でのデータ数が一致していない可能性

### 1.3 コードの問題
- **エラーハンドリング不足**: ファイルが存在しない場合の処理がない
- **グローバル変数依存**: `figPCA`関数内で未定義の`y`変数を参照
- **可視化機能の不完全**: PCA可視化やSVM境界表示が動作しない可能性
- **訓練・推論モード切り替えの煩雑さ**: 複数箇所のコメントアウト切り替えが必要
- **モデル保存機能なし**: 訓練済みモデルの保存・読み込み機能がない

### 1.4 課題要件との乖離
- **人間の顔画像データが必要**: 現在は動物の顔のみ
- **アプリケーション化**: ボーナス点のためのGUI/Webアプリが未実装
- **4ページレポート**: 学術論文形式のレポートテンプレートがない

---

## 2. 必要な修正・追加事項

### 2.1 環境整備 【優先度: 高】

#### A. requirements.txt作成
```
numpy>=1.21.0
scikit-learn>=1.0.0
Pillow>=8.0.0
matplotlib>=3.5.0
opencv-python>=4.5.0
streamlit>=1.0.0  # Webアプリ用（オプション）
```

#### B. 仮想環境設定手順
```bash
python -m venv facial_classification_env
facial_classification_env\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2.2 データセット拡充 【優先度: 高】

#### A. 人間の顔データセット収集
- **推奨データセット**:
  - CelebA dataset（有名人の顔）
  - LFW (Labeled Faces in the Wild)
  - 自作データセット（許可済み人物写真）

#### B. データディレクトリ構造変更
```
af_data/
├── human_faces/          # 人間の顔データ（メイン）
│   ├── cat_like/         # 猫っぽい人間の顔
│   ├── dog_like/         # 犬っぽい人間の顔
│   ├── fox_like/         # キツネっぽい人間の顔
│   └── tiger_like/       # トラっぽい人間の顔
└── reference_animals/    # 参考用動物の顔
    ├── cat/
    ├── dog/
    ├── fox/
    └── tiger/
```

#### C. データ増強
- 各カテゴリ最低100枚以上を目標
- 回転、反転、明度調整による画像増強 (余力あれば)

### 2.3 コード構造改善 【優先度: 高】

#### A. 訓練・推論機能の分離

**1. 訓練関数の作成**
```python
def train_model():
    """訓練モード - PCAとSVMモデルを訓練"""
    print("訓練モードを開始...")
    x, y = imgLoad()
    if x is None or y is None:
        print("データの読み込みに失敗")
        return
    
    # モデル訓練
    acc, pca, svm = xSVM(x, y)
    
    # オプション：可視化
    figPCA(x, y, eigen=True, mean=True, com=100)
    figSVM(x, y)
    
    print(f"訓練完了。精度: {acc}")
    return pca, svm
```

**2. 推論関数の作成**
```python
def inference_mode():
    """推論モード - 事前訓練済みモデルを読み込んで評価"""
    print("推論モードを開始...")
    try:
        pca = joblib.load('./pca_110.pkl')
        svm = joblib.load('./svm_c1.5_com110.pkl')
        imgEva()
    except FileNotFoundError as e:
        print(f"モデルファイルが見つかりません: {e}")
        print("先に訓練モードを実行してください")
```

**3. メイン関数の簡素化**
```python
if __name__ == "__main__":
    # 1行のコメントアウトでモード切り替え
    
    # 訓練モード - この行のコメントを外す
    # train_model()
    
    # 推論モード - この行のコメントを外す
    inference_mode()
```

#### B. 関数パラメータ修正

**1. figPCA関数修正**
```python
def figPCA(x, y=None, eigen=False, mean=False, com=100):
    pca = PCA(n_components=com)
    x_pca = pca.fit_transform(x)
    print("保持率:", sum(pca.explained_variance_ratio_))
    
    if eigen:
        eigen1 = pca.components_[0].reshape((128, 128))
        eigen2 = pca.components_[1].reshape((128, 128))
        plt.subplot(1, 2, 1)
        plt.imshow(eigen1, cmap='gray')
        plt.title('固有ベクトル 1')
        plt.subplot(1, 2, 2)
        plt.imshow(eigen2, cmap='gray')
        plt.title('固有ベクトル 2')
        plt.show()
    
    if mean and y is not None:
        labels = ['cat', 'dog', 'fox', 'tiger']
        colors = ['red', 'green', 'blue', 'orange']
        for i in range(4):
            plt.scatter(
                x_pca[y == i, 0],
                x_pca[y == i, 1],
                label=labels[i],
                color=colors[i],
                alpha=0.6,
                edgecolor='k'
            )
        plt.legend()
        plt.show()
    
    return x_pca, pca
```

**2. figSVM関数修正**
```python
def figSVM(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    x_train_2d, pca_vis = figPCA(x_train, y_train, False, False, 2)
    svm_vis = SVC(kernel='rbf', C=10, gamma='scale')
    svm_vis.fit(x_train_2d, y_train)
    
    # 可視化作成
    h = 1.0
    x_min, x_max = x_train_2d[:, 0].min() - 1, x_train_2d[:, 0].max() + 1
    y_min, y_max = x_train_2d[:, 1].min() - 1, x_train_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    Z = svm_vis.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    labels = ['cat', 'dog', 'fox', 'tiger']
    colors = ['red', 'green', 'blue', 'orange']
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    for i in range(4):
        plt.scatter(
            x_train_2d[y_train == i, 0],
            x_train_2d[y_train == i, 1],
            c=colors[i],
            label=labels[i],
            edgecolor='k',
            alpha=0.8
        )
    plt.legend()
    plt.title('SVM決定境界 (2D PCA)')
    plt.show()
```

**3. accTest関数修正**
```python
def accTest(x, y, lowN=120, highN=121, stepN=1, lowC=2, highC=3, stepC=1):
    name = str(lowN) + '_' + str(highN) + '_' + str(stepN)
    txt = []
    for i in range(lowN, highN, stepN):
        for j in range(int(lowC / stepC), int(highC / stepC), 1):
            a, _, _ = xSVM(x, y, i, j * stepC)
            res = str(a) + ' > components = ' + str(i) + ', C = ' + str(j * stepC) + '\n'
            print(res)
            txt.append(res)
    with open('./res' + str(name) + '.txt', 'w') as file:
        file.writelines(txt)
```

### 2.4 基本的なエラーハンドリング 【優先度: 中】

#### A. 基本的なエラーハンドリング追加
```python
def imgLoad():
    try:
        images = []
        labels = []
        for id, animal in enumerate(['cat', 'dog', 'fox', 'tiger']):
            class_path = './af_data/' + str(animal)
            if not os.path.exists(class_path):
                print(f"警告: ディレクトリ {class_path} が見つかりません")
                continue
            
            files = os.listdir(class_path)
            if len(files) == 0:
                print(f"警告: {class_path} にファイルがありません")
                continue
                
            for i in files:
                try:
                    img = Image.open(class_path + '/' + i).convert('L')
                    img = img.resize((128, 128))
                    imgArr = np.array(img).flatten()
                    images.append(imgArr)
                    labels.append(id)
                except Exception as e:
                    print(f"エラー: {class_path}/{i} の読み込みに失敗: {e}")
                    continue
        
        if len(images) == 0:
            print("画像の読み込みに成功しませんでした")
            return None, None
            
        return np.array(images), np.array(labels)
    except Exception as e:
        print(f"imgLoadでエラー: {e}")
        return None, None
```

### 2.5 ドキュメント整備 【優先度: 中】

#### A. README.md更新
- プロジェクト概要
- インストール手順
- 使用方法（訓練vs推論）
- データセット情報

#### B. コードコメント
- 各関数の明確な説明コメント追加
- 訓練・推論モード切り替えの説明

---

## 3. 実装優先順位

### Phase 1: 基盤整備（Week 1）
1. 環境構築（requirements.txt、仮想環境）
2. 人間の顔データセット収集・整理
3. コード構造改善（訓練・推論関数分離）
4. 基本的なエラーハンドリング

### Phase 2: テスト・改良（Week 2）
1. 収集した人間の顔データでテスト
2. 残存するバグの修正
3. 必要に応じてパラメータ最適化
4. ドキュメント作成

### Phase 3: 完成・最適化（Week 3）
1. Webアプリケーション開発（ボーナス用）
2. レポート作成
3. 最終テスト・デバッグ
4. プレゼンテーション準備

---

## 4. 推奨チーム分担

- **Project Lead**: 全体進捗管理、コード統合
- **Data Collector**: 人間の顔データセット収集・前処理
- **Data Analyst**: PCA/SVM最適化、テスト
- **App Developer**: Webアプリケーション開発（ボーナス用）
- **Report Writer**: 学術論文形式レポート作成
- **Presenter**: プレゼンテーション準備・発表

---

## 5. 成功指標

### 課題要件
- 4ページ学術論文完成
- チーム貢献明記
- 明確な構造のコード提出
- 期日内提出
- プレゼンテーション実施

### ボーナス(時間が許せば)
- Webアプリケーション動作
- ユーザーフレンドリーなUI
- リアルタイム分類機能 