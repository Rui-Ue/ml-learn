
# README

<br>

機械学習全般をしっかり勉強し直す．理論と実装．
参考資料探し．


<br>
<br>

# 方針

<br>

ML全般の学習とNN系(ディープラーニング含む)の学習は，分けて考えた方が良さそう．資料も別になってることが多い．

軸となる本を決めて，必要に応じて他の資料も見る感じが良さそう．摘み食いもあり．

もちろんweb上の資料やサーベイ論文も全然あり．

軸としたい本の要件としては...
- 綺麗な流れがあって体系的
- 理論がある程度は書いてある(長い目で見て)
- なるべく最近の情勢が考慮されてる

他に把握したい点としては...
- 手法のメリットデメリット・使い所
- python 等での実装例


大学に入れないので一旦手元の本を使うしかなさそう．もしくは１日だけ取りに行く．あるいは，pdfでオープンにされてる物とか，電子書籍(特にオライリーならpdfで売ってる)とか．

ゼミ後追記．
神嶌先生が運営している朱鷺の杜(ときのもり)の[この記事](http://ibisforest.org/index.php?Book)の[この記事](http://ibisforest.org/index.php?%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92#TextBook)，ML系の本の紹介と，書評がある．


<br>
<br>

# ML全般

<br>

## [統計的学習の基礎（ESL）](https://www.kyoritsu-pub.co.jp/bookdetail/9784320123625)

[原著は公開](https://web.stanford.edu/~hastie/ElemStatLearn/)されてる．訳本は研究室にあって一部pdf．

理論や流れがガチ．これを軸にするなら，摘み食いしつつ，実装は他の本を頼る．

Google の [TJO さんもオススメ](https://tjo.hatenablog.com/entry/2020/02/03/190000)
> Deep Learning以降の流れをカバーしていないのでDeep全盛の現在では物足りないと思う人も多いかもしれませんが、それ以外のほぼ全ての機械学習分野の話題がカバーされているので辞書として使う上では今でも最適の鈍器です。意外かもですが、Kaggleでは全員が使うといっても差し支えないGDBT系のモデルのアルゴリズムとその解説をきちんと載せている数少ない書籍の一つです。

<br>

## [パターン認識と機械学習（PRML）](https://www.maruzen-publishing.co.jp/item/b294524.html)

[原著は公開](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)されてて，訳本は研究室．

Microsoft Research の人が書いていて，ESLと同様に理論はガチ．やるなら摘み食い．

ESLと違ってこちらはベイズベースな感じ．人気なのはこっち．
[両者の比較](https://www.quora.com/Which-book-is-more-accessible-to-a-CS-student-in-machine-learning-the-Elements-of-Statistical-Learning-or-Pattern-Recognition-And-Machine-Learning
)も参考になりそう．
> ESL has more of a statistician’s perspective, while PRML has more of a computer scientist’s perspective

<br>

## [統計的機械学習の数理100問 with Python](https://www.kyoritsu-pub.co.jp/bookdetail/9784320125070)

まだ発売されていないが，趣旨は好き．
> 前者(ESL)は事典に近く，読者が何かを身につけるために書かれた書籍ではない。後者(James本)は初心者を対象として，感覚的な理解を促してパッケージを使わせることに終始し，本質に近づく視点が欠如していると言わざるを得ない。本書を読むことで，機械学習に関する知識が得られることはもちろんだが，脳裏に数学的ロジックを構築し，プログラムを構成して具体的に検証していくという，データサイエンス業界で活躍するための資質が得られる。

<br>

## [Understanding Machine Learning: From Theory to Algorithms](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/index.html)

ゼミ後に追記．齋藤さんが教えてくれた．
> PRMLやESLよりも評判のよいと網羅的な書籍です

風間さん出身の東大佐藤一誠先生のオススメ．pdfが無料公開されてる．ダウンロードした．

<br>

## 実装の資料

### [Kaggleで勝つデータ分析の技術](https://gihyo.jp/book/2019/978-4-297-10843-4)

タイトルの割に結構良さそう．以前に買って手元にあり．実装はもちろん手法のメリットデメリットの説明もあり．
Google の [TJO さんもオススメ](https://tjo.hatenablog.com/entry/2020/02/03/190000#%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%81%AE%E5%AE%9F%E8%B7%B52020-Feb-04%E8%BF%BD%E8%A8%98)
> タイトルだけ見るとKaggleに特化したかのような内容に見えますが、実際にはKaggleのみならずありとあらゆる機械学習の実践の場で問題となる事項が網羅され尽くしています。即ち評価指標の置き方・特徴量の扱い方・モデル評価と交差検証の方法・モデルのチューニング・モデルの組み合わせ方・leakageのような落とし穴、などなど僕が常日頃その重要性を説く"ML design"の考え方が本当に「全て」載っており、機械学習の実務家であれば必携の書と言って良いかと思います。

### [ソフトバンク加藤さんの本](https://www.shoeisha.co.jp/book/detail/9784798155654)

理論は薄いが幅広い手法の典型的な実装例がある．辞書的に使うなら良さそう．著者に興味あってKindleで買ってた．これも[オススメ](https://www.shoeisha.co.jp/book/detail/9784798155654)されている．

### [オライリーscikit-learn本](https://www.oreilly.co.jp/books/9784873117980/)

scikit-learnをメインにした本で，実装が網羅的に乗ってる．辞書的に使えるかも．





<br>
<br>

# NN系(ディープラーニング含む)

<br>

## [ゼロから作るDeepLearning](https://www.oreilly.co.jp/books/9784873117584/)

これは決定．手元にあるしpdfでも買える．この本を軸に進めて，理論的に気になった詳細は他で埋める．

<br>

## [深層学習](https://www.kspub.co.jp/book/detail/1529021.html)

研究室の借りて家にある．理論がまとまってる．辞書的に使えば良いかも．

<br>

## DNNライブラリについて

[これ](https://www.atmarkit.co.jp/ait/articles/1910/31/news028.html)によると，産業界では TensorFlow が人気で，学術界では PyTorch が人気．

とりあえず集中講義でやったレベルの Pytorch までは身につけときたい．書籍というか web でドキュメントや記事読むのが良いのかも．


<br>
<br>

# その他

<br>

## [Python実践データ分析100本ノック](https://www.shuwasystem.co.jp/book/b497338.html)

相当リアルなので良さそう．Pandas レベルなので既に知ってるものが多いかもしれないけど．

<br>

## [MLPシリーズ](https://www.kspub.co.jp/book/series/S043.html)

これっていう研究課題が決まった後に読むと良いかも．理論系はもちろん，様々な応用分野(画像認識,音声認識,自然言語,Webデータ)別の書籍が出ている．