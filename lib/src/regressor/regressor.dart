import 'package:ml_linalg/matrix.dart';
import 'package:reservoir/src/regressor/ridge.dart';

abstract class Regressor {
  Matrix get weights;
  bool get isFlatten;
  factory Regressor.fit(List<List<double>> trainX, Object trainY,
      {double lambda = 1.0, RegressorType type = RegressorType.ridge}) {
    switch (type) {
      case RegressorType.ridge:
        return RidgeRegressor.fit(trainX, trainY, lambda: lambda);
    }
  }
  factory Regressor.loadParams(String paramsJson,
      {RegressorType type = RegressorType.ridge}) {
    switch (type) {
      case RegressorType.ridge:
        return RidgeRegressor.fromJson(paramsJson);
    }
  }
  Object predict(List<List<double>> X);
  String saveParams();
}

enum RegressorType { ridge }
