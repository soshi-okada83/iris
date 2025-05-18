MLPClassifier を用いたアヤメ（Iris）データセットの分類
このプロジェクトでは、scikit-learn の MLPClassifier（多層パーセプトロン）を使用して、アヤメの品種を特徴量から分類するニューラルネットワークを実装しています。モデルの学習、評価、性能の可視化を含みます。

・使用データセット
sklearn.datasets に含まれる Iris データセットを使用しています。

150 件のサンプル

各サンプルの特徴量（4つ）

花弁の長さ（cm）

花弁の幅（cm）

がく片の長さ（cm）

がく片の幅（cm）

3つのクラス（品種）

Setosa（セトサ）

Versicolor（バージカラー）

Virginica（バージニカ）

・モデル構成
多層パーセプトロン（MLP）を以下の条件で使用しています:

隠れ層：1層（10ユニット）

活性化関数：ReLU

最適化手法：Adam

反復：1000

・必要なライブラリ
以下のライブラリをインストールしてください
pip install numpy pandas matplotlib seaborn scikit-learn

・実行方法
以下のコードをターミナルにてコピペしてください
python iris.py
※スクリプトのファイル名が異なる場合は適宜読み替えてください

・出力内容
学習データに対する精度（Train Accuracy）の表示

予測ラベルと実際のラベルの表示（テストデータに対して）

損失関数の推移（Loss Curve）のグラフ表示

混同行列（Confusion Matrix）のヒートマップ表示

・補足
データセットは 80% を訓練用、20% をテスト用に分割しています

------English------

Iris Flower Classification using MLPClassifier
This project implements a simple neural network using scikit-learn's MLPClassifier to classify Iris flower species based on their features. It includes model training, evaluation, and visualization of performance.

・Dataset
The Iris dataset from sklearn.datasets is used, which contains

150 samples

4 features per sample

Petal length (cm)

Petal width (cm)

Sepal length (cm)

Sepal width (cm)

3 classes

Setosa

Versicolor

Virginica

・Model
A Multi-layer Perceptron (MLP) classifier is used:

One hidden layer with 10 neurons

Activation function: ReLU

Optimizer: Adam

iterations: 1000

・Requirements
Make sure you have the following libraries installed:
pip install numpy pandas matplotlib seaborn scikit-learn

・How to Run
Please copy and paste the following code into the terminal.
python iris.py
Replace iris.py with your script filename.

・Output
Train Accuracy – The accuracy of the model on the training set.

Predicted vs Actual Labels – A printout of the model predictions compared to actual values from the test set.

Loss Curve – A plot showing the decrease in loss function during training.

Confusion Matrix – A heatmap that visualizes the model's performance.

・Notes
The model is trained using 80% of the data and tested on the remaining 20%.
