import 'dart:convert';

import 'package:ml_linalg/linalg.dart';
import 'package:ml_reservoir/src/activations/activations.dart';
import 'package:ml_reservoir/src/reservoir/reservoir.dart';
import 'package:ml_reservoir/src/utils/matrix_utils.dart';

class ESNModule implements ReservoirModule {
  @override
  final double alpha;
  final Matrix _w;
  final Matrix _wIn;
  final int maxLogLength;
  @override
  final Matrix Function(Matrix) activation;
  @override
  final List<Matrix> stateLogs = []; //状態のログ
  ESNModule(
    int inputDim, {
    int outputDim = 64,
    this.alpha = 0.5,
    double connectionProb = 0.05,
    double spectralRadius = 0.99,
    this.activation = sigmoid,
    Matrix? wIn,
    Matrix? w,
    this.maxLogLength = -1,
  })  : _wIn = wIn ?? randomGaussianMatrix(inputDim, outputDim),
        _w = w ?? _createWIn(outputDim, connectionProb: connectionProb);

  factory ESNModule.fromJson(String json) {
    final jsonMap = jsonDecode(json);
    final wIn = Matrix.fromList(jsonMap['w_in'] as List<List<double>>);
    final wRes = Matrix.fromList(jsonMap['w_res'] as List<List<double>>);
    return ESNModule(0, wIn: wIn, w: wRes);
  }

  @override
  String saveWeights() {
    final jsonMap = {
      'w_in': _wIn.toList(),
      'w_res': _w.toList(),
      'max_log_length': maxLogLength,
    };
    return jsonEncode(jsonMap);
  }

  @override
  List<List<List<double>>> call(List<List<List<double>>> X) {
    final transposedX = transpose3dList(X, 1, 0, 2);

    //時間を進める
    for (var t = 0; t < transposedX.length; t++) {
      next(transposedX[t], isNeedState: false);
    }
    return transpose3dList(
      stateLogs.map((e) => e.map((e) => e.toList()).toList()).toList(),
      1,
      0,
      2,
    );
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

  @override
  List<List<List<double>>> next(List<List<double>> xT,
      {bool isNeedState = true}) {
    final currentState = stateLogs.isEmpty
        ? zerosMatrix(xT.length, _w.columnCount)
        : stateLogs.last; //前の状態を取得
    final input = Matrix.fromList(xT); //B,X
    //input,output
    final uin = input * _wIn;
    final xres = currentState * _w;
    final nextState =
        currentState * (1 - alpha) + activation(uin + xres) * alpha;
    if (maxLogLength > 0 && stateLogs.length >= maxLogLength) {
      stateLogs.removeAt(0);
    }
    stateLogs.add(nextState);
    if (isNeedState) {
      return transpose3dList(
        stateLogs.map((e) => e.map((e) => e.toList()).toList()).toList(),
        1,
        0,
        2,
      );
    }
    return [];
  }

  @override
  void resetState() {
    stateLogs.clear();
  }
}
