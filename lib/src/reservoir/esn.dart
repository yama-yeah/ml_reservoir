import 'package:ml_linalg/linalg.dart';
import 'package:reservoir/src/activations/activations.dart';
import 'package:reservoir/src/reservoir/reservoir.dart';
import 'package:reservoir/src/utils/matrix_utils.dart';

class ESNModule implements ResovoirModule {
  @override
  final int inputDim;
  @override
  final int outputDim;
  @override
  final double alpha;
  final Matrix _w;
  final Matrix _wIn;
  @override
  final Matrix Function(Matrix) activation;
  ESNModule(
    this.inputDim, {
    this.outputDim = 64,
    this.alpha = 0.5,
    double connectionProb = 0.05,
    double spectralRadius = 0.99,
    this.activation = sigmoid,
  })  : _wIn = randomGaussianMatrix(inputDim, outputDim),
        _w = _createWIn(outputDim);

  @override
  List<List<List<double>>> call(List<List<List<double>>> X) {
    final transposedX = transpose3dList(X, 1, 0, 2);
    List<List<List<double>>> stateLogs = []; //状態のログ
    //時間を進める
    for (var t = 0; t < transposedX.length; t++) {
      final currentState = t == 0
          ? zerosMatrix(X.length, outputDim)
          : Matrix.fromList(stateLogs.last); //前の状態を取得
      final input = Matrix.fromList(transposedX[t]); //B,X
      //input,output
      final uin = input * _wIn;
      final xres = currentState * _w;
      final next = currentState * (1 - alpha) + activation(uin + xres) * alpha;
      stateLogs.add(next.toList().map((e) => e.toList()).toList());
    }
    return transpose3dList(stateLogs, 1, 0, 2);
  }

  static Matrix _createWIn(int outputDim,
      {double connectionProb = 0.05, double spectralRadius = 0.99}) {
    Matrix w = randomGaussianMatrix(outputDim, outputDim);
    final randomMask = randomMatrix(outputDim, outputDim)
        .mapElements((element) => element < connectionProb ? 1.0 : 0.0);
    w = w.multiply(randomMask); //結合度からスパースにする

    final wEigenValues = w.eigvals(maxIterations: 1000);
    double maxEigenValue = wEigenValues.abs().max();
    if (maxEigenValue == 0) {
      maxEigenValue = 0.0001;
    }
    return w / maxEigenValue * spectralRadius;
  }
}
