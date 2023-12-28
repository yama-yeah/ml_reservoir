import 'package:ml_linalg/linalg.dart';
import 'dart:math' as math;

extension MatrixUtils on Matrix {
  Matrix inverse() {
    final Matrix eye = Matrix.identity(columns.length);
    return solve(eye);
  }

  Vector eigvals({int maxIterations = 100, double tolerance = 1e-10}) {
    assert(isSquare);
    int n = columnCount;
    Matrix A = this;
    //Matrix V = Matrix.identity(n);
    Matrix aPrev;

    for (int i = 0; i < maxIterations; i++) {
      aPrev = A;
      final qr = _qrDecompositionGramSchmidt(A);
      Matrix Q = Matrix.fromList(qr.$1);
      Matrix R = Matrix.fromList(qr.$2);

      A = R * Q;
      //V = V * Q;

      Matrix diff = A - aPrev;
      double chebyshevNorm = diff.mapElements((element) => element.abs()).max();
      if (chebyshevNorm < tolerance) {
        break;
      }
      final isTriangular =
          A.any((element) => element.any((element) => element == tolerance));

      if (isTriangular) {
        break;
      }
    }

    List<double> eigenvalues = List.generate(n, (i) => A[i][i]);
    //List<Matrix> eigenvectors = List.generate(n, (i) => V.column(i));

    return Vector.fromList(eigenvalues);
  }

  (List<List<double>>, List<List<double>>) _qrDecompositionGramSchmidt(
      Matrix A) {
    assert(isSquare);
    final dim = columnCount;
    List<List<double>> Q = List.generate(dim, (_) => List.filled(dim, 0.0));
    List<List<double>> R = List.generate(dim, (_) => List.filled(dim, 0.0));

    for (int k = 0; k < A.columnCount; k++) {
      Vector a = A.getColumn(k);
      Vector u = A.getColumn(k);

      for (int i = 0; i < k; i++) {
        Vector qI = Vector.fromList(Q[i]);
        double projectionScale = a.dot(qI);
        R[i][k] = projectionScale;
        u -= qI * projectionScale;
      }

      double normU = u.norm();

      R[k][k] = normU;
      if (normU == 0) {
        Q[k] = u.toList();
      } else {
        Q[k] = (u / normU).toList();
      }
    }
    for (int i = 0; i < dim; i++) {
      for (int j = i + 1; j < dim; j++) {
        double temp = Q[i][j];
        Q[i][j] = Q[j][i];
        Q[j][i] = temp;
      }
    }
    return (Q, R);
  }
}

double _boxMuller(double mean, double std) {
  final u1 = RandomRef().random.nextDouble();
  final u2 = RandomRef().random.nextDouble();
  final z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2);
  return z0 * std + mean;
}

Matrix randomGaussianMatrix(int rows, int columns,
    {double mean = 0.0, double std = 1.0}) {
  final List<List<double>> data = [];
  for (var i = 0; i < rows; i++) {
    final List<double> row = [];
    for (var j = 0; j < columns; j++) {
      row.add(_boxMuller(mean, std));
    }
    data.add(row);
  }
  return Matrix.fromList(data);
}

Matrix randomMatrix(int rows, int columns,
    {double min = 0.0, double max = 1.0}) {
  final random = RandomRef().random;
  final List<List<double>> data = [];
  for (var i = 0; i < rows; i++) {
    final List<double> row = [];
    for (var j = 0; j < columns; j++) {
      row.add(random.nextDouble() * (max - min) + min);
    }
    data.add(row);
  }
  return Matrix.fromList(data);
}

Matrix zerosMatrix(int rows, int columns) {
  final List<List<double>> data = [];
  for (var i = 0; i < rows; i++) {
    final List<double> row = [];
    for (var j = 0; j < columns; j++) {
      row.add(0.0);
    }
    data.add(row);
  }
  return Matrix.fromList(data);
}

List<List<List<double>>> transpose3dList(
    List<List<List<double>>> X, int axis0, int axis1, int axis2) {
  final List<List<List<double>>> result = [];
  //numpy.transposeのように、axis0, axis1, axis2の順番で転置する
  Map<int, int> lengthMap = {0: X.length, 1: X[0].length, 2: X[0][0].length};
  for (var i = 0; i < lengthMap[axis0]!; i++) {
    final List<List<double>> row = [];
    for (var j = 0; j < lengthMap[axis1]!; j++) {
      final List<double> col = [];
      for (var k = 0; k < lengthMap[axis2]!; k++) {
        Map<int, int> indexMap = {axis0: i, axis1: j, axis2: k};
        for (var l = 0; l < 3; l++) {
          assert(indexMap.containsKey(l));
        }
        col.add(X[indexMap[0]!][indexMap[1]!][indexMap[2]!]);
      }
      row.add(col);
    }
    result.add(row);
  }
  return result;
}

//singleton
class RandomRef {
  static final RandomRef _instance = RandomRef._internal();
  factory RandomRef() => _instance;
  RandomRef._internal();
  final random = math.Random(43);
}
