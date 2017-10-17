# filters
画像と内積する行列。画像＊フィルタ＝特徴マップ。
フィルタの数は、2の階乗がスタンダード

# ストライド
フィルタを一度にずらす間隔

# Pooling
画像をぼかす層
画像の一定の範囲にフィルタをかけ、画像をぼかす（例：Maxpooling…フィルタ内の一番大きい数値を出力する）

# Dropout
ネットワーク内のノードのうち、いくつかを無効化する
ネットワークの自由度を強制的に縮小させ、過学習を防止

---
[1] https://deepage.net/deep_learning/2016/11/07/convolutional_neural_network.html
[2] http://forums.fast.ai/t/my-dogs-vs-cats-models-always-have-0-5-accuracy-whats-wrong/1665/14
[3] https://keras.io/ja/preprocessing/image/
