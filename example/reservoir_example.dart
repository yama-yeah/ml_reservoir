import 'dart:math';

import 'package:ml_linalg/linalg.dart';
import 'package:ml_reservoir/reservoir.dart';

void main() {
  final List<List<List<double>>> data = [];
  for (var i = 0; i < 512; i++) {
    final List<List<double>> T = [];
    for (var j = 0; j < 64; j++) {
      final t = [sin(pi * j / 64) + cos(pi * j / 64 + pi / 512 * i)];
      T.add(t);
    }
    data.add(T);
  }
  data.shuffle();
  final trainData = data.sublist(0, 256);
  final testData = data.sublist(256, 512);
  final esn = ReservoirModule.newNetwork(1, outputDim: 32);
  final trainOutput = esn(trainData);
  esn.resetState();
  final testOutput = esn(testData);
  //trainOutputの次元が、BatchSize,TimeStep,OutputDimなので、BatchSize,TimeStep*OutputDimに変換する
  final List<List<double>> trainX = [];
  for (var i = 0; i < trainOutput.length; i++) {
    final List<double> row = [];
    for (var j = 0; j < trainOutput[i].length; j++) {
      row.addAll(trainOutput[i][j]);
    }
    trainX.add(row);
  }
  //trainDataの次元が、BatchSize,TimeStep,1なので、BatchSize,TimeStepに変換する
  final List<List<double>> trainY = [];
  for (var i = 0; i < trainData.length; i++) {
    final List<double> row = [];
    for (var j = 0; j < trainData[i].length; j++) {
      row.add(trainData[i][j][0]);
    }
    trainY.add(row);
  }

  final ridge = Regressor.fit(trainX, trainY);
  final List<List<double>> testX = [];
  for (var i = 0; i < testOutput.length; i++) {
    final List<double> row = [];
    for (var j = 0; j < testOutput[i].length; j++) {
      row.addAll(testOutput[i][j]);
    }
    testX.add(row);
  }
  final predY = ridge.predict(testX) as List<List<double>>;
  final mae = (Matrix.fromList(predY) -
              Matrix.fromList(
                  testData.map((e) => e.map((e) => e[0]).toList()).toList()))
          .map((e) => e.map((e) => e.abs()).toList())
          .toList()
          .reduce((value, element) => value + element)
          .reduce((value, element) => value + element) /
      predY.length;
  print('MAE: $mae');
}
