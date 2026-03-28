# クラスタリングと次元削減

正解ラベルなしで、データの内在的な構造を発見する。

```mermaid
graph TD
    subgraph クラスタリング["クラスタリング: 似たものをまとめる"]
        KM[K-Means<br/>セントロイドベース]
        DB[DBSCAN<br/>密度ベース]
    end
    subgraph 次元削減["次元削減: 本質を抽出する"]
        PCA_[PCA<br/>線形・高速]
        TSNE[t-SNE<br/>非線形・可視化]
    end

    style KM fill:#3498db,color:#fff
    style DB fill:#9b59b6,color:#fff
    style PCA_ fill:#e67e22,color:#fff
    style TSNE fill:#e74c3c,color:#fff
```

---

## K-Means

### アルゴリズム

```mermaid
graph LR
    Init["① K個のセントロイドを<br/>ランダムに初期化"] --> Assign["② 各点を最も近い<br/>セントロイドに割り当て"]
    Assign --> Update["③ 各クラスタの平均を<br/>新しいセントロイドに"]
    Update --> Check{"収束?"}
    Check -->|No| Assign
    Check -->|Yes| Done["完了"]

    style Init fill:#3498db,color:#fff
    style Done fill:#2ecc71,color:#fff
```

### EMアルゴリズムとしての解釈

```
目的関数: J = Σₖ Σ_{x∈Cₖ} ‖x - μₖ‖²
```

| ステップ | 操作 | EMとの対応 |
|:---:|---|:---:|
| **割り当て** | μを固定してクラスタを最適化 | E-step |
| **更新** | クラスタを固定してμを最適化 | M-step |

各ステップでJは単調に減少し収束する。ただし**局所最適解**に陥る可能性がある。

### 限界

```mermaid
graph TD
    subgraph "K-Meansが得意"
        A["⬤ 球状クラスタ"]
    end
    subgraph "K-Meansが苦手"
        B["🌙 三日月形"]
        C["📏 サイズが不均一"]
    end
```

- クラスタ数Kを事前に指定する必要がある
- 球状のクラスタしか検出できない
- 外れ値に敏感

---

## DBSCAN

### アイデア：密度でクラスタを定義する

```mermaid
graph TD
    subgraph "3種類の点"
        Core["● コア点<br/>eps内にmin_samples個以上"]
        Border["◐ ボーダー点<br/>コア点のeps内にいる"]
        Noise["○ ノイズ点<br/>どちらでもない"]
    end

    Core --> |"密度到達可能な<br/>コア点をつなぐ"| Cluster["クラスタ"]
    Border --> Cluster
    Noise --> Outlier["外れ値 &#40;-1&#41;"]

    style Core fill:#2ecc71,color:#fff
    style Border fill:#f39c12,color:#fff
    style Noise fill:#95a5a6,color:#fff
```

### K-Meansとの比較

| | K-Means | DBSCAN |
|:---:|:---:|:---:|
| **クラスタ数** | 事前指定 | 自動検出 |
| **クラスタ形状** | 球状のみ | 任意の形状 |
| **ノイズ** | 全点をクラスタに割り当て | ノイズ点を除外 |
| **パラメータ** | K | eps, min_samples |

---

## 主成分分析 (PCA)

### アイデア：分散が最大の方向を見つける

```mermaid
graph LR
    subgraph "元のデータ (2D)"
        A["● ● ●<br/>● ● ● ●<br/>● ● ●"]
    end
    subgraph "PCA後 (1D)"
        B["●●●●●●●●●●"]
    end
    A -->|"分散最大の方向に<br/>射影"| B
```

### アルゴリズム

```mermaid
graph TD
    A["① データを中心化<br/>&#40;平均を引く&#41;"] --> B["② 共分散行列を計算<br/>C = &#40;1/n&#41; XᵀX"]
    B --> C["③ 固有値分解<br/>Cw = λw"]
    C --> D["④ 固有値が大きい順に<br/>k個の固有ベクトルを選択"]
    D --> E["⑤ 射影<br/>Z = X × W"]

    style A fill:#e67e22,color:#fff
    style E fill:#2ecc71,color:#fff
```

### なぜ固有ベクトルか

分散最大化問題をラグランジュ乗数法で解くと：

```
max wᵀCw  subject to ‖w‖ = 1
    ↓
Cw = λw  （固有値問題）
```

**固有ベクトル** = 分散最大の方向、**固有値** = その方向の分散量。

### 寄与率

```
寄与率ₖ = λₖ / Σλᵢ
```

「第k主成分がデータの分散の何%を説明するか」。累積寄与率90%を目安にk個を選ぶ。

---

## t-SNE

### PCAとの違い

```mermaid
graph TD
    subgraph PCA
        A1["線形変換"]
        A2["全体構造を保存"]
        A3["高速"]
        A4["逆変換可能"]
    end
    subgraph t-SNE
        B1["非線形"]
        B2["局所構造を保存"]
        B3["低速"]
        B4["可視化専用"]
    end

    style PCA fill:#e67e22,color:#fff
    style t-SNE fill:#e74c3c,color:#fff
```

### 核心的なアイデア

```mermaid
graph TD
    H["高次元での類似度<br/>ガウス分布で定義<br/>p_ij"] --> KL["KLダイバージェンスを<br/>最小化"]
    L["低次元での類似度<br/>Student-t分布で定義<br/>q_ij"] --> KL
    KL --> Opt["勾配降下法で<br/>低次元座標を更新"]

    style KL fill:#e74c3c,color:#fff
```

### なぜ Student-t分布か（Crowding問題）

高次元では「中程度の距離」の点が大量にある。これを低次元に押し込むと近くに密集してしまう。

```
ガウス分布 (高次元)     Student-t分布 (低次元)
    ╱╲                       ─╲
   ╱  ╲                    ╱   ╲
  ╱    ╲                  ╱     ╲
 ╱      ╲                ╱       ╲
╱        ╲              ╱     裾が重い → 遠い点にも
                              ゼロでない確率を割り当て
```

Student-t分布の**重い裾**が、遠い点を低次元でも遠く配置することを許容する。

### Perplexity

σᵢ は各点ごとに異なり、perplexity パラメータで制御する。

| Perplexity | 効果 |
|:---:|---|
| **小** (5〜10) | 局所構造重視。小さなクラスタが見える |
| **大** (30〜50) | 大域構造重視。大きな構造が見える |

二分探索で各点の適切な σᵢ を決定する。
