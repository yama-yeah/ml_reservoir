import 'package:ml_linalg/linalg.dart';

abstract class ResovoirModule {
  int get inputDim;
  int get outputDim;
  double get alpha;
  Matrix Function(Matrix) get activation;
  ResovoirModule(
    inputDim, {
    outputDim = 64,
    alpha = 0.5,
    connectionProb = 0.05,
    spectralRadius = 0.99,
  });
  List<List<List<double>>> call(List<List<List<double>>> X);
}
