import 'dart:convert';

import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
import 'package:reservoir/src/regressor/regressor.dart';

class RidgeRegressor implements Regressor {
  @override
  final Matrix weights;
  @override
  final bool isFlatten;
  const RidgeRegressor._(this.weights, this.isFlatten);
  factory RidgeRegressor.fit(List<List<double>> trainX, Object trainY,
      {double lambda = 1.0}) {
    final List<List<double>> trainXCp = [];
    for (var i = 0; i < trainX.length; i++) {
      trainXCp.add([...trainX[i], 1.0]);
    }
    final matrixX = Matrix.fromList(trainXCp);
    final lambdaI = Matrix.identity(trainXCp[0].length) * lambda;
    final isFlatten = trainY is List<double>;
    final trainYmathlable = isFlatten
        ? Vector.fromList(trainY)
        : Matrix.fromList(trainY as List<List<double>>);
    final xT = matrixX.transpose();
    final Matrix w =
        ((xT * matrixX + lambdaI).inverse() * xT * trainYmathlable);

    return RidgeRegressor._(w, isFlatten);
  }
  factory RidgeRegressor.fromJson(String json) {
    final jsonMap = jsonDecode(json);
    final w = Matrix.fromList(jsonMap['w'] as List<List<double>>);
    final isFlatten = jsonMap['isFlatten'] as bool;
    return RidgeRegressor._(w, isFlatten);
  }
  @override
  String saveParams() {
    final jsonMap = {'w': weights.toList(), 'isFlatten': isFlatten};
    return jsonEncode(jsonMap);
  }

  @override
  Object predict(List<List<double>> X) {
    X = X.map((e) => [...e, 1.0]).toList();
    final Matrix matrixX = Matrix.fromList(X);
    return isFlatten
        ? (matrixX * weights).toVector().toList()
        : (matrixX * weights).toList().map((e) => e.toList()).toList();
  }
}
