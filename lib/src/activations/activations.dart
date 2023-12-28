import 'package:ml_linalg/matrix.dart';

Matrix sigmoid(Matrix x) {
  return (x.exp() + 1).mapElements((element) => 1 / element);
}
