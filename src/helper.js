import * as tf from "@tensorflow/tfjs";

// data: csv file with x,y values per line
// Returns data needed for statter plot
// an array of objects with x, y key
// [{x:1, y: 5}, {x:2,y:10}]
export const forScatterFromFile = data => {
  const vals = data.split("\n").map(val => {
    const [x, y] = val.split(",");
    if (x && y) {
      return { x: parseFloat(x), y: parseFloat(y) };
    }
  });
  return vals.splice(1, vals.length - 2); // undefined at end. Not sure why...
};

// x: array of inputs [34,23,67]
// y: array of results, paired with 'x' [56, 45, 80]
// Returns data needed for statter plot
// an array of objects with x, y key
// [{x:1, y: 5}, {x:2,y:10}]
export const forScatter = (x, y) => {
  const results = x.reduce((acc, curr, i) => {
    acc.push({ x: curr, y: y[i] });
    return acc;
  }, []);

  return results;
};

// X: 1d Tensor
// Y: 1d Tensor
// Returns data needed for statter plot
// an array of objects with x, y key
// [{ x: 1, y: 5 }, { x: 2, y: 10 }];
export const forScatterFromTensor = (X, Y) => {
  let x = Array.from(X.dataSync());
  let y = Array.from(Y.dataSync());

  const results = x.reduce((acc, curr, i) => {
    acc.push({ x: curr, y: y[i] });
    return acc;
  }, []);

  return results;
};

export const getXandYFromFile = data => {
  return data.split("\n").reduce(
    (acc, val) => {
      const [x, y] = val.split(",");
      if (x) acc.x.push(x);
      if (y) acc.y.push(y);
      return acc;
    },
    { x: [], y: [] }
  );
};

// Returns predicted output
export const yHat = (x, model) => {
  let [a, b] = model;
  const X = tf.tensor1d([x]);
  a = tf.scalar(a);
  b = tf.scalar(b);

  // [191.8968811, 75.2018051, 61.695034, ..., 84.0904922, 84.4796982, 72.8036728]
  // ignore-prettier
  return a
    .mul(X)
    .add(b)
    .dataSync();
};

export const yHats = (x, model) => {
  const predictions = x.map(x => yHat(x, model));

  // [191.8968811, 75.2018051, 61.695034, ..., 84.0904922, 84.4796982, 72.8036728]
  return Array.from(predictions);
};

// x: array of inputs [34,23,67]
// y: array of results, paired with 'x' [56, 45, 80]
// Returns data needed for statter plot
export const train = (x, y) => {
  const X = tf.tensor1d(x);
  const Y = tf.tensor1d(y);

  // denominator = X.dot(X) - X.mean()*X.sum()
  const denominator = X.dot(X).sub(X.mean().mul(X.sum()));

  // const a = (X.dot(Y) - Y.mean() * X.sum()) / denominator;
  const a = X.dot(Y)
    .sub(Y.mean().mul(X.sum()))
    .div(denominator);

  // const b = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y)) / denominator;
  const b = Y.mean()
    .mul(X.dot(X))
    .sub(X.mean().mul(X.dot(Y)))
    .div(denominator);

  // [1.9726121674845978, 8644240756601382] Not a Tensor
  return [Array.from(a.dataSync()).pop(), Array.from(b.dataSync()).pop()];
};
