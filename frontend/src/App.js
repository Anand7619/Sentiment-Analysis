
import React, { useState } from "react";
import axios from "axios";

function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);

  const handlePredict = async () => {
    const res = await axios.post("http://localhost:8000/predict", { text });
    console.log(res.data);
    setResult(res.data);
  };

  return (
    <div style={{ padding: "2rem" }}>
      <h2>Sentiment Analysis</h2>
      <textarea rows={5} cols={60} onChange={(e) => setText(e.target.value)} />
      <br />
      <button onClick={handlePredict}>Predict</button>
      {result && (
        <div>
          <p><strong>Label:</strong> {result.label}</p>
          <p><strong>Score:</strong> {result.score.toFixed(3)}</p>
        </div>
      )}
    </div>
  );
}

export default App;
