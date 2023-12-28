import 'package:ml_linalg/matrix.dart';

abstract class Regressor {
  Matrix get weights;
  bool get isFlatten;
  Regressor.fit(Matrix trainX, Matrix trainY, {double lambda = 1.0});
  Object predict(List<List<double>> X);
}
