import 'package:ml_linalg/linalg.dart';
import 'package:reservoir/src/reservoir/esn.dart';

abstract class ReservoirModule {
  ///状態の更新率
  double get alpha;

  ///活性化関数
  Matrix Function(Matrix) get activation;

  ///状態のログ
  List<Matrix> get stateLogs;

  ///
  ///入力の次元と出力の次元を指定してモジュールを作成する
  ///connectionProbは結合度を表す。0.05ならば5%のニューロンが結合する
  ///spectralRadiusはスペクトル半径がどれだけ1に近づくかを表す　基本的に１に近ければ近いほど良いが、計算誤差があるのでそこも考慮する
  factory ReservoirModule.newNetwork(
    int inputDim, {
    int outputDim = 64,
    double alpha = 0.5,
    double connectionProb = 0.05,
    double spectralRadius = 0.99,
    ReservoirType type = ReservoirType.esn,
  }) {
    switch (type) {
      case ReservoirType.esn:
        return ESNModule(
          inputDim,
          outputDim: outputDim,
          alpha: alpha,
          connectionProb: connectionProb,
          spectralRadius: spectralRadius,
        );
    }
  }

  factory ReservoirModule.loadNetwork(String weightsJson,
      {ReservoirType type = ReservoirType.esn}) {
    switch (type) {
      case ReservoirType.esn:
        return ESNModule.fromJson(weightsJson);
    }
  }

  ///モジュールの重みを文字列に書き出す
  String saveWeights();

  ///与えられたTに沿って状態を返す
  ///XはB,T,Xの3次元配列
  List<List<List<double>>> call(List<List<List<double>>> X);

  ///リアルタイムで状態を更新する
  ///XはB,Xの2次元配列
  List<List<List<double>>> next(List<List<double>> xT);

  ///状態を初期化する
  void resetState();
}

enum ReservoirType {
  ///Echo State Network
  esn,
}
