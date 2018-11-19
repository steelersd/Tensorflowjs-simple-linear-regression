import React from "react";
import ReactDOM from "react-dom";
import "./styles.css";
import data from "./data_1d.csv";
import * as tf from "@tensorflow/tfjs";
import { train, getXandYFromFile, forScatter, yHats } from "./helper";
import SimpleRegression from "./SimpleRegression";

const { x, y } = getXandYFromFile(data);
const model = train(x, y);

const Yhats = yHats(x, model);
const sampleData = forScatter(x, y);
const predictionData = forScatter(x, Yhats);

function App() {
  return (
    <div className="App">
      <h1>Simple linear regression</h1>
      <SimpleRegression sampleData={sampleData} predictionData={predictionData} />
    </div>
  );
}

const rootElement = document.getElementById("root");
ReactDOM.render(<App />, rootElement);
