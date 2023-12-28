import 'package:ml_linalg/linalg.dart';
import 'package:reservoir/reservoir.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

void main() {
  final data = getIrisDataFrame().dropSeries(names: ['Id']);
  final X = data.dropSeries(names: ['Species']).toMatrix();
  final List<List<double>> trainX = [];

  for (var i = 0; i < X.rows.length; i++) {
    trainX.add(X.toList()[i].toList());
  }
  final y = data['Species'];
  Set<String> uniqueSpecies = Set<String>.from(y.data);
  final yVector = Vector.fromList(
      y.data.map((e) => uniqueSpecies.toList().indexOf(e).toDouble()).toList());
  final model = RidgeRegressor.fit(trainX, yVector.toList());
  final predY = model.predict(trainX) as List<double>;
  final mae =
      Vector.fromList(predY).distanceTo(yVector, distance: Distance.manhattan) /
          yVector.length;
  print('MAE: $mae');
  final dynamic a = 1;
  print(a as int);
}
