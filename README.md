## 機械学習PoCと本番運用のあいだ

#### はじめに

機械学習を使ったPoC(Proof of Concept; 効果検証)は、近頃のAIブームを受けて盛んに行われていることと思います。そしてその後、(とても幸運なことに、)PoCで良い結果を得られた場合、PoCの仕組みを本番運用したいという話が持ち上がります。しかしながら、探索的データ分析や予測モデルの作成などは多くの知見が共有されている一方、それらを運用していくための知見はまだあまり多くないように思います。

本記事では、PoCから本番運用へと移行するあいだに、技術的には何が必要なのかを検討していきます。機械学習PoCが一過性のまま終わらず、本番運用によって価値を生み出していくための一助となれば幸いです。

#### 本記事で書くこと

- PoCにおけるデータ分析の進め方
- **機械学習PoCのテスト運用の進め方(本記事のメイントピック)**
- **PoC、テスト運用の各フェーズでのアーキテクチャ(本記事のメイントピック)**
- 本番運用に向けて追加で検討すべきこと

本記事では特にテスト運用にフォーカスします。テスト運用フェーズでは運用と分析を並列で行うことが多いですが、分析をしながらどのようにシステムのアーキテクチャをアップデートして本番運用に近づけていくかの一例を記載しています。

##### 本記事で書かないこと
- 予測精度向上に関する詳細
  - 探索的データ分析の詳細
  - 前処理や特徴量生成の詳細
  - 予測モデルの詳細
- ミドルウェア(データベースやWebサーバ)よりも低レイヤーに関して
- 機械学習PoCでのコンサルテーションに関するスキル

コンサルテーションに関するスキルは、不確実なことの多い機械学習プロジェクトにおいて大変重要な点ですが、本記事では技術にフォーカスするため記載しません。

#### 本記事で利用するデータ

- [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data) のデータを使用する
- 説明を完結にするため、「application_train.csv」のみを使用する
- 説明のために、データを以下のように分割して使用する。

本記事では、Kaggleで過去に開催されたコンペティションである「[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data)」のデータを使用します。このコンペティションでは個人の与信情報をもとに、各個人が債務不履行になるかどうかを予測するものです。データはローンの申込みごとにレコードがあり、1つのレコードには申込者の与信情報と、その人が返済できたか、あるいは債務不履行になったかを示すラベルが含まれています。

この記事では、自分がとあるローン貸付会社のデータ分析部門にいて、この与信情報をもとに、与信判定を機械学習を用いて自動化するとしたらどうするか、という仮定のもとで話を進めていきます。

この仮定をした上で、説明の都合上、本コンペティションで利用できるデータのうち、「application_train.csv」を図のように分割して利用します。分割したデータはそれぞれ以下を仮定しています。

- 「initial.csv」：過去の与信情報。PoCを始める際に提供されたデータ
- 「20201001.csv」：2020年10月分の与信情報。テスト運用ではこのデータは「initial.csv」と合わせて訓練データとして扱う
- 「20201101.csv」：2020年11月分の与信情報。テスト運用ではこのデータは「initial.csv」と合わせて訓練データとして扱う
- 「20201201.csv」：2020年12月分の与信情報。テスト運用ではこのデータは「initial.csv」と合わせて訓練データとして扱う
- 「20210101.csv」：2021年1月分の与信情報。テスト運用ではこの月から予測を始める

![split data](/Users/koyajima/Code/between_poc_and_production/images/split_data.png)

データを分割したい実際のコードは以下です。

**split_data.ipynb**

```python
# データ読み込み
train = pd.read_csv('../data/rawdata/application_train.csv')

# initial.csv出力
train.iloc[:len(train)-40000,:].to_csv('../data/poc/initial.csv', index=False)

# 各月データ出力
train.iloc[len(train)-40000:len(train)-30000,:].to_csv('../data/poc/20201001.csv', index=False)
train.iloc[len(train)-30000:len(train)-20000,:].to_csv('../data/poc/20201101.csv', index=False)
train.iloc[len(train)-20000:len(train)-10000,:].to_csv('../data/poc/20201201.csv', index=False)
train.iloc[len(train)-10000:,:].to_csv('../data/poc/20210101.csv', index=False)
```

#### 検討に際して、想定する状況

本記事では、説明をしやすくするため、以下のようなプロジェクトを想定して検討を進めていきます。以降のストーリーは、今回使用する「Home Credit Default Risk」のデータから筆者が妄想したものであるため、現実の会社や業務とは全く無関係です。また、筆者は与信業務に関して全くの素人であり、現実の業務と異なる可能性が大いにあります。

---

銀行の与信業務を自動化するプロジェクトに参画している。
与信業務は、現在は審査部門で人手で行われているが、工数の低減や与信判定の精度向上のために機械学習が使えないか検討している。
サンプルデータはすでに提供されており、PoCを行う段階にある。サンプルデータには過去にローンを貸りた人が債務不履行になったかどうかが含まれている。このデータをもとに、もし新しくローンを借りたい人が来たときに、その人が債務不履行になるかどうかを予測することで、ローンを貸し付けるかどうかを判定できるようにしたい。

#### 本記事の対象とするプロジェクトのスコープ

機械学習プロジェクトは通常、企画、PoC、テスト運用、本番運用と続いていきます。本記事では、技術的なポイントにフォーカスするため、PoCからテスト運用までをスコープとして記載します。特にテスト運用については、本番運用に移行するために多くの検証が必要なため、3つのフェーズに分けて説明します。なお本番運用については、筆者の経験が浅いため、検討すべき点を挙げるだけに留めています。

![scope](/Users/koyajima/Code/between_poc_and_production/images/scope.png)

#### 本記事で想定する体制

本記事では、プロジェクトをスモールスタートで進めていく方針で記載するため、ミニマムな体制を想定します。具体的には、ビジネス部門(本記事では審査部門)をやり取りをするコンサルタントが1名、データ分析からシステム構築までを行うデータサイエンティストが1名になります。実際には、監督としてマネージャーがいると思いますが、本記事では登場しません。またステークホルダーとしては、ビジネス部門が存在します。

![organization](/Users/koyajima/Code/between_poc_and_production/images/organization.png)

### PoCフェーズ
##### このフェーズでの目的

このフェーズの目的は、与信判定の自動化が可能なのかを検証することです。そこで、本フェーズでは主に2つのポイントについて検証していきます。1つは、機械学習によって与信判定はどれくらいの精度で予測できるのかを検証すること、もう1つは利用するデータが本番運用時にも利用できるものであるか、(具体的には予測時に利用可能なデータなのか、レコード同士に関連性がないか、)データの妥当性を検証することです。

##### このフェーズでのアーキテクチャ

このフェーズではJupyterLabだけで作業を進めていきます。

アーキテクチャ図;JupyterLabのみ

#### データの妥当性検証

データサイエンティストとしてはすぐにでもデータを見始めたいところですが、まずはそのデータの妥当性を検証します。もしデータに欠陥があれば、そのデータを利用した予測は無駄になってしまう可能性が高いからです。妥当性の検証では、主に2つのポイントを確かめます。1つは、各レコードにおいて、レコード内のそれぞれの列のデータはいつ入手できるのかを明らかにすることです。各列のデータは、こちらに提供された時点では揃っているように思えますが、それらが同時に入手できるとは限りません。最も簡単な例では、目的変数である「債務不履行に陥ったかどうか」が分かるのは、他の列より後になるでしょう。もう1つの確認ポイントは、レコード同士に関係がないかをチェックすることです。例えば、ある人物が2回ローンの申し込みをしたとして、学習データに1回目のレコードが、テストデータに2回目のレコードが存在してしまうと、予測に有利に働いてしまうことが予想できます。このような場合には、両方のレコードを、学習データまたはテストデータのどちらかに含めるようにするなどの処理をします。これらのポイント以外にも、そもそも各列のデータは何を意味しているのか、レコードの単位は何なのか(今回のデータでは人単位なのか、ローン申請単位なのか)などをビジネス部門からヒアリングして、データの定義を明確にしておくことも大切です。データの列ごとにこれらの確認ポイントをチェックする表を作成すると良いかもしれません。

#### 探索的データ分析
データの妥当性が検証できたら、(あるいは妥当性の検証と並行して、)サンプルデータを可視化しながら、どのような列(特徴量)があるのかを見ていきます。この作業によってデータを理解し、特徴量エンジニアリングやモデル選択時の参考にします。また、データに問題がないかを探すきっかけとしても有用です。

まずは列ごとに、データ型、欠損値の割合などを確認します。

**eda.ipynb**

```python
"""
各コードの実行結果は「>」以下に示します。
また、説明に不要と思われる一部の実行結果は、
「...」「省略」と記載して省略しています。
"""

# ライブラリ読み込み
import pandas as np

# データ読み込み
train = pd.read_csv('../../data/rawdata/application_train.csv')

# 概要
train.head()
>
省略

# 行数、列数
f"行:{len(train)} / 列:{len(train.columns)}"
>
'行:307511 / 列:122'

# データ型
train.dtypes
>
SK_ID_CURR                        int64
TARGET                            int64
NAME_CONTRACT_TYPE               object
...

# 欠損値の数と割合
for c in train.columns:
    print(f'{c}  数:{train[c].isnull().sum()} / 割合:{train[c].isnull().sum() / len(train) * 100:0.4f}%')
>
SK_ID_CURR  数:0 / 割合:0.0%
...
NAME_TYPE_SUITE  数:1292 / 割合:0.4201%
...
COMMONAREA_AVG  数:214865 / 割合:69.8723%
...
```

次に分布を見るために、可視化をしてみます。
データ型が数値型の場合はヒストグラムを、文字列型の場合はバーチャートを使います。

**eda.ipynb**

```python
# ライブラリ読み込み
from bokeh.plotting import figure, output_file, show, output_notebook
from bokeh.models import ColumnDataSource, Grid, LinearAxis, Plot, VBar, HoverTool

# グラフをJupyterLab上で出力する設定
output_notebook()

# ヒストグラム
def plot_histogram(series, title, width=1000):
    p = figure(plot_width=width, plot_height=400, title=title, toolbar_location=None, tools="")
    hist, edges = np.histogram(series, density=True, bins=30)
    p.quad(
        top=hist,
        bottom=0,
        left=edges[:-1],
        right=edges[1:],
        fill_color="navy",
        alpha=0.2
    )
    p.y_range.start = 0
    show(p)

# バーチャート
def plot_bar_chart(series, title, width=1000):
    items = dict(series.value_counts())
    keys = list(items.keys())
    values = list(items.values())
    source = ColumnDataSource(data=dict(
    x=keys,
    y=values,
    ))
    TOOLTIPS = [
    ("列名", "@x"),
    ("カウント", "@y"),
    ]
    p = figure(plot_width=width, plot_height=400, x_range=keys, title=title,
               toolbar_location=None, tooltips=TOOLTIPS, tools="")
    glyph = VBar(x="x", top="y", width=0.9)
    p.add_glyph(source, glyph)
    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    show(p)
    
# 列ごとにグラフを出力する。欠損値は「-999」で補完する
for c in train.columns:
    series = train[c].copy()
    if series.dtype == 'int64' or series.dtype == 'float64':
        series.fillna(-999, inplace=True)
        plot_histogram(series, c)
    else:
        series.fillna('-999', inplace=True)
        plot_bar_chart(series, c)
```

出力されたグラフの例は以下です。

**AMT_CREDIT**

![graph_hist](/Users/koyajima/Code/between_poc_and_production/images/graph_hist.png)

**NAME_INCOME_TYPE**

![graph_bar](/Users/koyajima/Code/between_poc_and_production/images/graph_bar.png)

#### 予測精度の検証

ここからは実際にモデルを作成して予測精度を検証します。精度検証の際に重要なのが評価指標ですが、今回は「Home Credit Default Risk」の評価指標と同じくROCのAUCを使います。実際はビジネス部門と協議して、どの指標を使うのかをあらかじめ合意しておきます。まずは、マニュアルで予測モデルを作る前に、「PyCaret」を使ってざっくりと予測してみます。これにより、どの特徴量・モデルが有効なのかを比較し、実際にモデルを作る際の参考にします。

**eda.ipynb**

```python
# ライブラリ読み込み
from pycaret.classification import *

# セットアップ
exp = setup(data=train,
            target='TARGET',
            session_id=123,
            ignore_features = ['SK_ID_CURR'],
            log_experiment = True,
            experiment_name = 'credit1', 
            n_jobs=-1,
            silent=True)
```

```python
# モデルごとの予測精度比較
models = compare_models(include=['lr', 'dt', 'rf', 'svm','lightgbm'],
                        sort='AUC')
```

![compare_model](/Users/koyajima/Code/between_poc_and_production/images/compare_model.png)

今回はPyCaretがデフォルトで用意しているモデルのうち、以下をピックアップして比較しています。

- ロジスティック回帰
- 決定木
- ランダムフォレスト
- SVM
- LightGBM

評価指標をAUCとすると、LightGBMが優秀のようにみえます。一般にLightGBMは多くの場合精度と実行速度の両面で優秀であることが多いです。ところで、このデータでは正例が少ない不均衡データのため、Recallがどのモデルでも小さくなっています。ビジネス上のゴールにもよりますが、この後のモデル構築では、例えばRecallを上げて貸し倒れをより防ぐモデルを作るかなどを検討します。本記事では、詳細なモデリングは行わず、以降ではLightGBMを使ってモデル作成をしていきます。

次はどの特徴量が有効なのかを調べるため、PyCaretでモデルをLightGBMのモデルを作成し、評価します。

```python
# LightGBMモデル作成
lgbm = create_model('lightgbm')
```

![create_model](/Users/koyajima/Code/between_poc_and_production/images/create_model.png)

```python
# モデルパラメータ詳細
evaluate_model(lgbm)
```

![evaluate_model](/Users/koyajima/Code/between_poc_and_production/images/evaluate_model.png)

```python
# SHAP重要度算出
interpret_model(lgbm)
```

![shap](/Users/koyajima/Code/between_poc_and_production/images/shap.png)

```python
# 重要度算出
importance = pd.DataFrame(models.feature_importances_, index=exp[0].columns, columns=['importance'])
importance.sort_values('importance', ascending=False)
```

![importance](/Users/koyajima/Code/between_poc_and_production/images/importance.png)

今回のデータのように特徴量が多い場合、予測で使用する特徴量を絞ることでモデルの安定性を保てる可能性が高まります。絞り方の簡単な方法としては、特徴量重要度を算出し、重要度が低いものを除外することです。今回は単純に、重要度上位の特徴量を使用します。なお、PyCaretが自動で前処理している列については、オリジナルの方の列を使うことにします。

それでは、マニュアルで予測モデルを作成していきます。

#### 前処理

前処理としては、簡単のため、今回は欠損値の補完のみを行います。

**forecast.ipynb**

```python
# 欠損値補完
train.fillna(-999, inplace=True)
```



#### 特徴量エンジニアリング

特徴量エンジニアリングとしては、特徴量の選択とカテゴリ変数のダミー変数化を行います。

**forecast.ipynb**

```python
# 特徴量選択
features = [
    'EXT_SOURCE_1',
    'EXT_SOURCE_3',
    'EXT_SOURCE_2',
    'DAYS_BIRTH',
    'AMT_CREDIT',
    'AMT_ANNUITY',
    'DAYS_ID_PUBLISH',
    'AMT_GOODS_PRICE',
    'DAYS_REGISTRATION',
    'DAYS_LAST_PHONE_CHANGE',
    'AMT_INCOME_TOTAL',
    'REGION_POPULATION_RELATIVE',
    'OWN_CAR_AGE',
    'AMT_REQ_CREDIT_BUREAU_YEAR',
    'HOUR_APPR_PROCESS_START',
    'TOTALAREA_MODE',
    'CODE_GENDER',
    'NAME_CONTRACT_TYPE',
    'NAME_EDUCATION_TYPE',
    'NAME_FAMILY_STATUS',
    'TARGET'
]
train = train.loc[:,features]

# カテゴリ変数をダミー化
train = pd.get_dummies(train, drop_first=True, prefix=['CODE_GENDER', 'NAME_CONTRACT_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS'], prefix_sep='_')
```



#### 予測
LightGBMを使ってモデルを作成します。また、[Optuna](https://github.com/optuna/optuna)を使ってハイパーパラメータのチューニングをします。

```python
# ラベルと特徴量を分離
target = 'TARGET'
X = train.drop(columns=target)
y = train[target]

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                   test_size=0.2,
                                                   random_state=0,
                                                   stratify=y)

# 訓練データを一部バリデーション用に分割
X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train,
                                                    test_size=0.2,
                                                    random_state=1,
                                                    stratify=y_train)

# LightGBM用のデータセット作成
categorical_features = []
lgb_train = lgb.Dataset(X_train, y_train,
                       categorical_feature=categorical_features,
                        free_raw_data=False)
lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train,
                       categorical_feature=categorical_features,
                       free_raw_data=False)

# パラメータチューニング
def objective(trial):

    param = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'n_jobs': -1,
        'boosting_type': 'gbdt',
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }

    evaluation_results = {}
    model = lgb.train(
        param,
        lgb_train,              
        num_boost_round=1000,
        valid_names=['train', 'valid'],    
        valid_sets=[lgb_train, lgb_eval],    
        evals_result=evaluation_results,     
        categorical_feature=categorical_features,
        early_stopping_rounds=50,               
        verbose_eval=10)        
    
    y_pred = model.predict(X_train, num_iteration=model.best_iteration)
    
    # metrics AUC
    fpr, tpr, thresholds = metrics.roc_curve(y_train, y_pred)
    score = metrics.auc(fpr, tpr)
    
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

print('Number of finished trials: {}'.format(len(study.trials)))
print('Best trial:')
trial = study.best_trial
print('  Value: {}'.format(trial.value))
print('  Params: ')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))
```

```python
# 二値分類
params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',     
         }
# チューニングしたパラメータを合体
params = dict(params, **study.best_params)

# 学習
evaluation_results = {}
model = lgb.train(
    param,
    lgb_train,              
    num_boost_round=1000,
    valid_names=['train', 'valid'],    
    valid_sets=[lgb_train, lgb_eval],    
    evals_result=evaluation_results,     
    categorical_feature=categorical_features,
    early_stopping_rounds=50,               
    verbose_eval=10)     

# 最もスコアが良いときのラウンドを保存
optimum_boost_rounds = model.best_iteration
```

```python
# 訓練データで予測
y_pred = model.predict(X_train, num_iteration=model.best_iteration)
fpr, tpr, thresholds = metrics.roc_curve(y_train, y_pred)
auc = metrics.auc(fpr, tpr)
print(auc)
plt.plot(fpr, tpr, label='ROC curve (area = %.2f)'%auc)
plt.legend()
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
```

![predict_train](/Users/koyajima/Code/between_poc_and_production/images/predict_train.png)

```python
# テストデータで予測
y_pred = model.predict(X_test, num_iteration=model.best_iteration)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
auc = metrics.auc(fpr, tpr)
print(auc)
plt.plot(fpr, tpr, label='ROC curve (area = %.2f)'%auc)
plt.legend()
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
```

![predict_test](/Users/koyajima/Code/between_poc_and_production/images/predict_test.png)

この予測精度の検証では、PyCaretを使用したときとほぼ同じ精度での予測ができました。現実ではこの結果からさらに深堀りした分析を行っていきますが、本記事ではここまでとします。

PoCの結果をビジネス部門に報告し、仮にこのPoCのシステムを本番運用していく方針になったと仮定して話を進めていきます。ただPoCからいきなり本番運用とはならず、実際は何回かのテスト運用を行い、徐々に本番運用に近づけていくはずです。そこで、以降では、テスト運用を3つのフェーズに分けて検討します。各フェーズで少しずつ機能を追加していき、運用の仕方が徐々に自動化され、本番運用に近づくようにします。

#### 補足：機械学習モデルの管理

TODO:MLflowの導入

### テスト運用

#### テスト運用の3つのフェーズ

PoCから本番運用に行くまでには運用の自動化や推論単体の実行など、いくつかの機能を実装する必要があります。しかし、必要な機能すべてをすぐに実装することは難しいと思います。さらに、この段階ではビジネス部門からの依頼でさらなる精度向上を求められていることでしょう。そこで、必要な機能を3つのフェーズに分けて徐々に実装することで、運用しながらも機能を拡充していくことにします。各フェーズではそれぞれ以下の機能を実装していきます。

1. データパイプラインの構築と運用の半自動化
2. 学習・推論APIの実装
3. クラウドへの移行と運用自動化

#### テスト運用フェーズ1：データパイプラインの構築と運用の半自動化

##### このフェーズでの目的

フェーズ1では、PoCで作成したシステムを一部自動化して運用できるようにします。その前に、PoCのプログラムを特徴量エンジニアリングや予測などのブロックに分割・整理して、データパイプラインを構築します。これにより、学習や推論を単体で実行したり、途中から再実行できるようにします。さらに、それぞれのブロックに分けたプログラムを自動実行・スケジューリング実行できるように、ワークフローエンジンであるAirflowを導入します。

##### このフェーズでのアーキテクチャ

PoCでは1つのJupyterNotebookのみを使っていましたが、このフェーズからは複数のNotebookに分割します。これらのNotebookを順番に実行できるように、新たに2つのOSSを導入します。1つは「[papermill](https://github.com/nteract/papermill)」で、このOSSはJupyterNotebookをコマンドラインから実行でき、かつ実行時にパラメータを渡すことができます。これにより、例えば実行する月をパラメータとして渡すことで、Notebookの中身を書き換えることなく異なる月の予測ができるようになります。さらに、各Notebookを順番に実行するために「[Airflow](https://airflow.apache.org/)」を使用します。このOSSは自動実行だけでなく、スケジューリング実行や成功通知・失敗通知など、運用自動化に便利な機能を備えています。

TODO；アーキテクチャ図;JupyterLab, papermill, Airflow

##### データパイプライン

PoCで作成したプログラムを、「データ蓄積」「特徴量エンジニアリング」「学習」「推論」の４つのブロックに分割します。ブロックに分けるときは、各ブロック同士はデータをインターフェースとすることで、疎結合になるようにします。これによって、プログラムのロジックの変更に伴う影響範囲を限定させます。各ブロックでは、プログラムのはじめにpapermillからパラメータとして実行月を渡せるように設定しておき、特定の月での実行ができるようにします。例えば、2021年1月分に運用する際のイメージは以下にようになります。

TODO：運用イメージ

以下には各ブロックのコードを記載します。基本的にはPoCで使用したプログラムの再利用で、運用自動化のために多少の追加・修正をしています。

##### 1. データ蓄積

```python
# パラメータ設定
TARGET_DATE = '20210101'
TARGET_DATE = str(TARGET_DATE)

# ライブラリ読み込み
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

# 初期データ読み込み
train = pd.read_csv('../../data/poc/initial.csv')

# 月次データ読み込み
INITIAL_DATE = '20201001'
date = datetime.strptime(INITIAL_DATE, '%Y%m%d')
target_date = datetime.strptime(TARGET_DATE, '%Y%m%d')

target_dates = []
while date < target_date:
    print(date)
    date_str = datetime.strftime(date, '%Y%m%d')
    target_dates.append(date_str)
    date += relativedelta(months=1)

# データ結合
monthly_dataframes = [train]
for d in target_dates:
    df = pd.read_csv(f'../../data/poc/{d}.csv')
    monthly_dataframes.append(df)

train = pd.concat(monthly_dataframes, axis=0)

# 前処理データ出力
train.to_pickle(f'../../data/trial/accumulate_{TARGET_DATE}.pkl')
```



##### 2. 特徴量エンジニアリング

```python
# パラメータ設定
TARGET_DATE = '20210101'
# DATA_TYPE = 'train'
DATA_TYPE = 'test'
TARGET_DATE = str(TARGET_DATE)

# ライブラリ読み込み
import pandas as pd

# データ読み込み
if DATA_TYPE == 'train':
    train = pd.read_pickle(f'../../data/trial/accumulate_{TARGET_DATE}.pkl')
elif DATA_TYPE == 'test':
    train = pd.read_csv(f'../../data/trial/{TARGET_DATE}.csv')

# 特徴量選択
features = ['EXT_SOURCE_1',
 'EXT_SOURCE_3',
 'EXT_SOURCE_2',
 'DAYS_BIRTH',
 'AMT_CREDIT',
 'AMT_ANNUITY',
 'DAYS_ID_PUBLISH',
 'AMT_GOODS_PRICE',
 'DAYS_REGISTRATION',
 'DAYS_LAST_PHONE_CHANGE',
 'AMT_INCOME_TOTAL',
 'REGION_POPULATION_RELATIVE',
 'OWN_CAR_AGE',
 'AMT_REQ_CREDIT_BUREAU_YEAR',
 'HOUR_APPR_PROCESS_START',
 'TOTALAREA_MODE',
 'CODE_GENDER',
 'NAME_CONTRACT_TYPE',
 'NAME_EDUCATION_TYPE',
 'NAME_FAMILY_STATUS',
'TARGET']
train = train.loc[:,features]

# 欠損値補完
# object型の場合はNAN、それ以外(数値型)は-999で補完する
for c in train.columns:
    if train[c].dtype == 'object':
        train[c].fillna('NAN', inplace=True)
    else:
        train[c].fillna(-999, inplace=True)

# カテゴリ変数をダミー化
train = pd.get_dummies(train, drop_first=True, prefix=['CODE_GENDER', 'NAME_CONTRACT_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS'], prefix_sep='_')

# 列がない場合は作成
feature_order = [
    'EXT_SOURCE_1',
    'EXT_SOURCE_3',
    'EXT_SOURCE_2',
    'DAYS_BIRTH',
    'AMT_CREDIT',
    'AMT_ANNUITY',
    'DAYS_ID_PUBLISH',
    'AMT_GOODS_PRICE',
    'DAYS_REGISTRATION',
    'DAYS_LAST_PHONE_CHANGE',
    'AMT_INCOME_TOTAL',
    'REGION_POPULATION_RELATIVE',
    'OWN_CAR_AGE',
    'AMT_REQ_CREDIT_BUREAU_YEAR',
    'HOUR_APPR_PROCESS_START',
    'TOTALAREA_MODE',
    'CODE_GENDER_M',
    'CODE_GENDER_XNA',
    'NAME_CONTRACT_TYPE_Revolving loans',
    'NAME_EDUCATION_TYPE_Higher education',
    'NAME_EDUCATION_TYPE_Incomplete higher',
    'NAME_EDUCATION_TYPE_Lower secondary',
    'NAME_EDUCATION_TYPE_Secondary / secondary special',
    'NAME_FAMILY_STATUS_Married',
    'NAME_FAMILY_STATUS_Separated',
    'NAME_FAMILY_STATUS_Single / not married',
    'NAME_FAMILY_STATUS_Unknown',
    'NAME_FAMILY_STATUS_Widow',
    'TARGET',
]
for c in feature_order:
    if c not in train.columns:
        train[c] = 0
      
# 列を並び替え
train = train.loc[:, feature_order]

# 特徴量データ出力
if DATA_TYPE == 'train':
    train.to_pickle(f'../../data/trial/feature_{TARGET_DATE}.pkl')
elif DATA_TYPE == 'test':
    train.to_pickle(f'../../data/trial/predict_{TARGET_DATE}.pkl')
```



##### 3. 学習

```python
# パラメータ設定
TARGET_DATE = '20210101'
TARGET_DATE = str(TARGET_DATE)

# ライブラリ読み込み
import pandas as pd
import numpy as np
import argparse
import shap
import optuna
import pickle
import lightgbm as lgb
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

# データ読み込み
train = pd.read_pickle(f'../../data/trial/feature_{TARGET_DATE}.pkl')

# ラベルと特徴量を分離
target = 'TARGET'
X = train.drop(columns=target)
y = train[target]

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                   test_size=0.2,
                                                   random_state=0,
                                                   stratify=y)

# 訓練データを一部バリデーション用に分割
X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train,
                                                    test_size=0.2,
                                                    random_state=1,
                                                    stratify=y_train)

# LightGBM用のデータセット作成
categorical_features = []
lgb_train = lgb.Dataset(X_train, y_train,
                       categorical_feature=categorical_features,
                        free_raw_data=False)
lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train,
                       categorical_feature=categorical_features,
                       free_raw_data=False)

# パラメータチューニング
def objective(trial):

    param = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'n_jobs': -1,
        'boosting_type': 'gbdt',
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }

    evaluation_results = {}
    model = lgb.train(
        param,
        lgb_train,              
        num_boost_round=1000,
        valid_names=['train', 'valid'],    
        valid_sets=[lgb_train, lgb_eval],    
        evals_result=evaluation_results,     
        categorical_feature=categorical_features,
        early_stopping_rounds=50,               
        verbose_eval=10)        
    
    y_pred = model.predict(X_train, num_iteration=model.best_iteration)
    
    # metrics AUC
    fpr, tpr, thresholds = metrics.roc_curve(y_train, y_pred)
    score = metrics.auc(fpr, tpr)
    
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

print('Number of finished trials: {}'.format(len(study.trials)))
print('Best trial:')
trial = study.best_trial
print('  Value: {}'.format(trial.value))
print('  Params: ')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))

# 二値分類
params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',     
         }
# チューニングしたパラメータを合体
params = dict(params, **study.best_params)

# 学習
evaluation_results = {}
model = lgb.train(
    param,
    lgb_train,              
    num_boost_round=1000,
    valid_names=['train', 'valid'],    
    valid_sets=[lgb_train, lgb_eval],    
    evals_result=evaluation_results,     
    categorical_feature=categorical_features,
    early_stopping_rounds=50,               
    verbose_eval=10)     

# 最もスコアが良いときのラウンドを保存
optimum_boost_rounds = model.best_iteration

# 訓練データで予測
y_pred = model.predict(X_train, num_iteration=model.best_iteration)
fpr, tpr, thresholds = metrics.roc_curve(y_train, y_pred)
auc = metrics.auc(fpr, tpr)
print(auc)
plt.plot(fpr, tpr, label='ROC curve (area = %.2f)'%auc)
plt.legend()
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)

# テストデータで予測
y_pred = model.predict(X_test, num_iteration=model.best_iteration)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
auc = metrics.auc(fpr, tpr)
print(auc)
plt.plot(fpr, tpr, label='ROC curve (area = %.2f)'%auc)
plt.legend()
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)

# モデル出力
pickle.dump(model, open(f'../../data/trial/model_{TARGET_DATE}.pkl', 'wb'))
```



#####  4. 推論

```python
# パラメータ設定
TARGET_DATE = '20210101'
TARGET_DATE = str(TARGET_DATE)

# ライブラリ読み込み
import pandas as pd
import numpy as np
import argparse
import shap
import optuna
import pickle
import lightgbm as lgb
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import pickle

# 予測用データ読み込み
test = pd.read_pickle(f'../../data/trial/predict_{TARGET_DATE}.pkl')

# モデル読み込み
model = pickle.load(open(f'../../data/trial/model_{TARGET_DATE}.pkl', 'rb'))

# 予測
test.drop(columns=['TARGET'], inplace=True)
y_pred = model.predict(test, num_iteration=model.best_iteration)
y_pred_max = np.round(y_pred)

# 予測データ作成
pred = pd.DataFrame()
# ID読み込み
rawdata = pd.read_csv(f'../../data/trial/{TARGET_DATE}.csv')
pred['ID'] = rawdata['SK_ID_CURR']
pred['TARGET'] = y_pred_max

# 予測データ出力
pred.to_csv(f'../../data/trial/pred_{TARGET_DATE}.csv', index=None)
```



##### 運用の半自動化

各処理が個別のプログラムに分割できたら、Airflowを利用して、それらを順序立てて実行できるようにします。実行時にパラメータとして予測月を渡すことで、その月の実行を行うことができます。また、「schedule_interval」にcron式でスケジューリング実行の日時を定義できます。以下に、2021年1月分を実行する場合のAirflowのコードを記載します。

```python
from datetime import timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

# from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago

args = {
    "owner": "admin",
}

dag = DAG(
    dag_id="trial_operation",
    default_args=args,
    schedule_interval="0 0 * * *",
    start_date=days_ago(2),
    dagrun_timeout=timedelta(minutes=60),
    # tags=[],
    params={'target_date': '20210101'},
)

accumulate = BashOperator(
    task_id="accumulate_train_data",
    bash_command="papermill --cwd ~/Code/between_poc_and_production/notebooks/trial/ ~/Code/between_poc_and_production/notebooks/trial/accmulate.ipynb ~/Code/between_poc_and_production/notebooks/trial/logs/accmulate.ipynb -p TARGET_DATE 20210101",
    dag=dag,
)

feat_eng_train = BashOperator(
    task_id="feature_engineering_train_data",
    bash_command="papermill --cwd ~/Code/between_poc_and_production/notebooks/trial/ ~/Code/between_poc_and_production/notebooks/trial/feature_engineering.ipynb ~/Code/between_poc_and_production/notebooks/trial/logs/feature_engineering.ipynb -p TARGET_DATE 20210101 -p DATA_TYPE train",
    dag=dag,
)

feat_eng_test = BashOperator(
    task_id="feature_engineering_test_data",
    bash_command="papermill --cwd ~/Code/between_poc_and_production/notebooks/trial/ ~/Code/between_poc_and_production/notebooks/trial/feature_engineering.ipynb ~/Code/between_poc_and_production/notebooks/trial/logs/feature_engineering.ipynb -p TARGET_DATE 20210101 -p DATA_TYPE test",
    dag=dag,
)

learn = BashOperator(
    task_id="learn",
    bash_command="papermill --cwd ~/Code/between_poc_and_production/notebooks/trial/ ~/Code/between_poc_and_production/notebooks/trial/learn.ipynb ~/Code/between_poc_and_production/notebooks/trial/logs/learn.ipynb -p TARGET_DATE 20210101",
    dag=dag,
)

inference = BashOperator(
    task_id="inference",
    bash_command="papermill --cwd ~/Code/between_poc_and_production/notebooks/trial/ ~/Code/between_poc_and_production/notebooks/trial/inference.ipynb ~/Code/between_poc_and_production/notebooks/trial/logs/inference.ipynb -p TARGET_DATE 20210101",
    dag=dag,
)

accumulate >> feat_eng_train >> learn

[learn, feat_eng_test] >> inference

if __name__ == "__main__":
    dag.cli()

```

Airflowでは定義したワークフローをフローチャートのように見ることができますが、上記のコードは以下のように可視化されます。一度実行したため、各ボックスが緑色(実行完了)になっています。

![airflow](/Users/koyajima/Code/between_poc_and_production/images/airflow.png)

このフェーズ1での実装によって、毎月の運用は以下のように自動化できました。

- PoCフェーズ
  1. 予測月のデータをアップロードする
  2. これまでの月の訓練データを結合する
  3. 訓練データを前処理・特徴量エンジニアリングする
  4. 訓練データからモデルを学習する
  5. テストデータを前処理・特徴量エンジニアリングする
  6. 学習したモデルを使ってテストデータを予測する
  7. 予測結果をダウンロードする
- テスト運用フェーズ1
  1. 予測月のデータをアップロードする
  2. Airflowからワークフローを実行する
  3. 予測結果をダウンロードする

#### テスト運用フェーズ2：学習・推論APIの実装

##### このフェーズでの目的

フェーズ1では、前処理や推論などの機能を個別のプログラムに分割し、papermillとAirflowを組み合わせることで、毎月の運用を大幅に自動化できました。このフェーズ2では、更に自動化を進めていきます。具体的には、フェーズ1で手作業としていたデータのアップロード・ダウンロードの実行、定形運用の実行、推論の個別実行について、それぞれのAPIとそれをGUIから実行できる画面を用意します。これにより、ビジネス部門をはじめとする非エンジニアのユーザでも簡単に運用ができるようにします。こうすることで、定型的な運用をユーザに任せることができ、エンジニアは開発作業により集中することができます。

##### このフェーズでのアーキテクチャ

フェーズ2では、新たにWebサーバを立て、それを操作できる画面を作ります。

アーキテクチャ図;JupyterLab, papermill, Airflow, FastAPI, React

##### Webサーバの作成

Webサーバには以下のAPIを用意します。

- 入力ファイルのアップロード機能
- 定形運用の実行機能
- 推論の実行機能
- 予測ファイルのダウンロード機能

今回は[FastAPI](https://github.com/tiangolo/fastapi) を利用してWebサーバを作成します。

```python
from fastapi import FastAPI
import subprocess
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel  # リクエストbodyを定義するために必要

app = FastAPI()


origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LearnParam(BaseModel):
    test: str


# curl -X POST http://localhost:8000/hello -H "Content-Type: application/json" -d "{'test':'hello'}"
@app.post("/hello")  # methodとendpointの指定
async def hello(param: LearnParam):
    print(param)
    # subprocess.call("airflow dags trigger test_dag".split(" "))
    return {"text": param}

```



##### GUIの作成

GUIではWebサーバのAPIを実行できるボタンと、データをアップロードできるフォームを用意します。今回は[React](https://github.com/facebook/react) と[Typescript](https://github.com/microsoft/TypeScript) を利用して自前で作成していますが、画面自体を作成するライブラリを用いたほうが素早く作成できるかもしれません。

```react
import React from 'react';
import logo from './logo.svg';
import './App.css';
import { makeStyles, createStyles, Theme } from '@material-ui/core/styles';
import Button from '@material-ui/core/Button';
import AppBar from '@material-ui/core/AppBar';
import { Toolbar } from '@material-ui/core';
import IconButton from '@material-ui/core/IconButton/IconButton';
import Typography from '@material-ui/core/Typography/Typography';

const useStyles = makeStyles((theme: Theme) =>
  createStyles({
    root: {
      '& > *': {
        margin: theme.spacing(1),
      },
    },
    menuButton: {},
  })
);

const App: React.FC = () => {
  const classes = useStyles();
  async function postData(url = '', data = {}) {
    // 既定のオプションには * が付いています
    const response = await fetch(url, {
      method: 'POST', // *GET, POST, PUT, DELETE, etc.
      // mode: 'cors', // no-cors, *cors, same-origin
      // cache: 'no-cache', // *default, no-cache, reload, force-cache, only-if-cached
      // credentials: 'same-origin', // include, *same-origin, omit
      headers: {
        'Content-Type': 'application/json',
      },
      // redirect: 'follow', // manual, *follow, error
      // referrerPolicy: 'no-referrer', // no-referrer, *no-referrer-when-downgrade, origin, origin-when-cross-origin, same-origin, strict-origin, strict-origin-when-cross-origin, unsafe-url
      body: JSON.stringify(data), // 本文のデータ型は "Content-Type" ヘッダーと一致する必要があります
    });
    return response.json(); // レスポンスの JSON を解析
  }

  return (
    <>
      <div className='App'>
        <AppBar position='static'>
          <Toolbar variant='dense'>
            <Typography variant='h6' color='inherit'>
              Between PoC and Production
            </Typography>
          </Toolbar>
        </AppBar>
      </div>
      <div className={classes.root}>
        <Button
          onClick={() =>
            postData('http://localhost:8000/hello', { test: 'hhhe' }).then(
              (data) => {
                console.log(data); // `data.json()` の呼び出しで解釈された JSON データ
              }
            )
          }
          variant='contained'
          color='primary'
        >
          Upload
        </Button>
      </div>
    </>
  );
};

export default App;

```

画面は以下のようなイメージになります。

TODO；フロントエンド画面のGIF

#### テスト運用フェーズ3：クラウドへの移行と運用自動化

##### このフェーズでの目的

フェーズ3では、さらなる運用の自動化に向けて、サーバなどの計算機をクラウドに移したり、一部の機能をマネージドサービスに移していきます。クラウドを使う目的は、インフラを

本記事では例としてAWSを利用します。


##### このフェーズでのアーキテクチャ

TODO：図EC2, S3, Lambda, SNS

各サーバをEC2上に構築し、

### 本番運用に向けて

この章では、本番運用に向けて検討すべきことをリストアップする

#### データの補完方法に関して

parquet,csv,pklの比較

#### データ連携に関して

審査部門とのデータのやり取りを自動化する

#### プログラムの再利用性の向上

Kubernetes、Dockerを導入する

#### クラウドのマネージドサービスの利用

GCP

AWS

AzureML

#### データの安全性について

男女差別がないか

#### データドリフト

分布が変わってないか、ラベルと特徴量の関係性は変わってないか

#### データのガバナンスについて

ログイン制御によるアクセス制御

#### 処理のスケーリング

Celery Executor

#### 本記事で利用したソフトウェアとそのバージョン

本記事で例として構築したシステムのソースは以下のGitHubのリポジトリに格納してあります。

https://github.com/koyaaarr/between_poc_and_production

| Software   | Version |
| ---------- | ------- |
| JupyterLab |         |
| Airflow    | 2.0     |
| MLflow     |         |
| Papermill  |         |
| FastAPI    |         |
| React      |         |
| Typescript |         |
| PyCaret    |         |
| pandas     |         |
| bokeh      |         |
|            |         |

#### 

### 参考文献

#### 機械学習プロジェクトの進め方に関して

- データ分析・AIのビジネス導入
- 仕事ではじめる機械学習

#### データ分析の技術に関して

- 前処理大全
- Kaggleで勝つデータ分析の技術
- 機械学習のための特徴量エンジニアリング

#### 機械学習PoCシステムの設計に関して

- データ指向アプリケーションデザイン
- Netflix
- リブセンス
- リクルート住まいカンパニー



