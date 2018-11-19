import React from "react";
import { Dot, ScatterChart, XAxis, YAxis, Tooltip, Legend, Scatter, CartesianGrid } from "recharts";

const SimpleRegression = props => {
  const { sampleData, predictionData } = props;
  return (
    <ScatterChart width={930} height={350} margin={{ top: 20, right: 20, bottom: 10, left: 10 }}>
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis type="number" dataKey="x" name="X" />
      <YAxis type="number" dataKey="y" name="Y" />
      <Tooltip cursor={{ strokeDasharray: "3 3" }} />
      <Legend />
      <Scatter name="Sample data" data={sampleData} fill="#8884d8" />
      <Scatter name="Model" data={predictionData} shape={<Dot />} fill="#336BFF" line />
    </ScatterChart>
  );
};

export default SimpleRegression;
