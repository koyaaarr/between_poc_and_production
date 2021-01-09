## 機械学習PoCと本番運用のあいだ

はじめに

機械学習を使ったPoC(Proof of Concept; 効果検証)は、近頃のAIブームを受けて盛んに行われていることと思います。その後、(とても幸運なことに、)PoCで良い結果を得られた場合、しばしばPoCの仕組みをこれから運用していきたいという話になっていきます。しかしながら、探索的データ分析や予測モデルの作成などは多くの知見が共有されている一方、それを運用していくための知見はまだあまり多くないように思います。

本記事では、PoCから本番運用へと移行するあいだに何が必要なのかを検討していきます。機械学習PoCが一過性のまま終わらず、本番運用によって価値を生み出していくための一助となれば幸いです。

本記事の対象とするプロジェクトのスコープ
構想策定
PoC←
テスト運用←
本番運用

本記事で想定する体制
コンサル1名、エンジニア1名
他ステークホルダー;クライアント

本記事で書かないこと
探索的データ分析の詳細
前処理や特徴量生成の詳細
予測モデルの詳細
ミドルウェアよりも低レイヤーに関して
コンサルテーションに関するスキル

本記事で利用するデータ
Santander

本記事で利用するライブラリ
JupyterLab
Airflow
MLflow

検討に際して、想定する状況
銀行の与信業務を自動化するプロジェクトに参画している。
現在与信業務は、審査部門で人手で行われているが、工数の低減や与信判定の精度向上のために機械学習が使えないか検討している。
サンプルデータはすでに提供されており、PoCを行う段階にある。

### PoCフェーズ
プロセス;PoC
アーキテクチャ図;JupyterLabのみ
このフェーズでは、機械学習によって与信判定がどれくらいの精度で予測できるのかを検証する。まずはサンプルデータを可視化しながら、どのような特徴量があるのかを見ていく。
＜探索的データ分析＞
分析の結果、特徴量XがN以下の場合は必ず与信が不可となっていることが分かった。このことを審査部門に問い合わせると、与信判定ではいくつかの基準に満たない場合は必ず不可にするルールがあると分かった。
このことから、与信判定の予測は2段階で行う方針となった。まず、基準を満たすかどうかをルールベースで判定し、その後、予測モデルによって与信判定をすることにした。
＜予測の方針＞
次に、予測対象となるデータを対象として、実際に予測モデルを構築する。まずは前処理を行い、その後特徴量を作成し、最後に予測を実施する。
＜前処理、特徴量作成＞
データのリークがないように、列、行ともに検討する
＜予測＞
予測の結果、与信判定は精度80％となり、まずまずの結果となった。誤り分析をしたところ、スコアが0.6以上の場合は精度が90%である一方、0.6以下の場合は精度が50%以下となることが分かった。そこで、スコアが0.6以下の場合は従来の人手による与信判定に切り替えることを提案した。その場合でも、全データの90%は機械による予測ができるため、当初の目標であるコスト低減は達成できる見込みである。

この結果を審査部門に報告したところ、ぜひ与信判定に導入したいとのフィードバックがあった。そこで、この与信判定自動化を検討することとなった。

### テスト運用

ユースケースを書く
このフェーズでは、与信判定の自動化を目指して、PoCで作成したシステムを一部自動化して運用し、本番運用に向けた課題を洗い出す。
まずはJupyterLabで行っていた処理を、前処理、特徴量作成、予測などのブロックに分割する。これには2つの狙いがある。1つはロジックの変更に伴う影響範囲を限定すること、もう1つは同じ処理の繰り返しがある場合にはJupyterNotebookに渡すパラメータを変えるだけでできるようにするためだ。後者の例として、正例、負例それぞれのデータに前処理を行う際に、それぞれのファイル名をJupyterNotebookに与えるだけで実現できるようにする。ブロックに分割するために、データパイプラインを考える。
＜データパイプライン＞
ブロックはファイルを界面にして分割し、各ブロックはファイルをインターフェイスとして、疎結合となるようにする。そして、各ブロックごとに、PoCのプログラムをベースにJupyterNotebookを作成する。
次に、一連のJupyterNotebookを自動で実行するために、ワークフローエンジンを導入する。本記事ではワークフローエンジンとしてAirflowを導入する。さらに、AirflowからJupyterNotebookを実行するために、Papermillを導入する。Papermillは、JupyterNotebookをBashから実行できるようにするライブラリで、実行時にJupyterNotebookにパラメータを渡すこともできる。
アーキテクチャ図;JupyterLab, papermill, Airflow
papermillから渡すJupyterNotebookに渡すパラメータは、定期運用時に必要な実行日時、繰り返し処理時に必要なファイル名などがある。
この改修によって、定期的にPoCのシステムを自動実行する仕組みができたため、テスト運用する準備ができた。
テスト運用してみると、審査部門から、与信判定をリアルタイムで実行したいという要望があった。また、学習は定期的に行うことでモデルをアップデートすべきという課題も発見した。

### テスト運用 ver2

このフェーズでは、与信判定をリアルタイムで実行する、学習を定期実行する仕組みを導入するなど、APIによって学習や推論を制御できるようにする。さらに、そのAPIをWeb画面から操作できるようにすることで、審査部門のユーザにも操作できるように改修する。
アーキテクチャ図;web server/web api/front end
学習や推論の実行はAirflowをBashから実行することで行えるようにする。

### テスト運用 ver3

このフェーズでは、与信判定をスケーラブルに行うために、システムをクラウド上に構築することを検討する。

### 本番運用に向けて

この章では、本番運用に向けて検討すべきことをリストアップする

### クラウドのマネージドサービスの利用

GCP

AWS

AzureML

#### データの安全性について

男女差別がないか

#### データドリフト

分布が変わってないか、ラベルと特徴量の関係性は変わってないか

### データのガバナンスについて

ログイン制御によるアクセス制御

#### 処理のスケーリング

Celery Executor


