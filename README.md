# NLPtutorial2020
小町研2020年度新入生の基礎勉強会として、NLPプログラミングチュートリアルを実施します。
教材URL：http://www.phontron.com/teaching.php?lang=ja
勉強会URL：http://cl.sd.tmu.ac.jp/groups/programming-tutorial

### 進め方（予定）
1. 勉強会で教材の内容を確認する。
2. 次週までに課題を解き、レポジトリにあげる。
3. 次週の冒頭で代表者がコードを説明する。
4. コードや内容についてTAが解説・レビューする。

### コーディングについて
基本的にPython3系を使用してください。
わからないところはTAまたは研究室の人に聞いてください。
他の人のコードは変更しないでください。

### レポジトリのあげかた
1. 各チュートリアル毎に、"tutorial##"（##はチュートリアルの番号を2桁で）フォルダを作成。
2. 資料内もしくは勉強会内で指定した形式で課題を解く。
3. `git branch tutorial<nn>/<name>`でブランチを作る
4. `git checkout tutorial<nn>/<name>`でブランチの中に移動する
5. `git add スクリプト名`（ワイルドカード*も使用可）
6. `git commit -m 'コメント'`（コメントは自由に）
7. `git pull`
8. `git push`
9. Pull Requestsを作ってレビューしてもらう

### みんなの進捗
![progress](https://github.com/tmu-nlp/NLPtutorial2020/blob/master/progress.png)


### Docker環境について
興味がある人はDocker環境で作業してみてください。
- Macの場合、brewで`docker`と`docker-compose`を入れましょう
- Windowsの場合、公式サイトにdocker desktop or windowsがあるので入れましょう

Makefileでコマンドを簡略化してあります。
1. docker環境の構築
`make docker-build`

2.1. docker環境内でコードを実行
`make docker-run FILE_NAME=./hirao/test.py`

2.2. docker環境内でjupyterを起動(使いたい人はどうぞ)
`make docker-run-jupyter PORT=12345`

Windowsの場合 (Macでもできる)
`docker-compose up -d`
でバックグラウンドで起動して
`docker exec -it nlp_turorial2020 bash`
とかで中に入って作業すると良さそう
