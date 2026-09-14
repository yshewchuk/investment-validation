import React from "react";
import ReactDOM from "react-dom/client";
import { App } from "./App";
import { createHttpDataClient } from "./api/client";
import "./styles.css";

const container = document.getElementById("root");
if (container === null) {
  throw new Error("missing #root element");
}

ReactDOM.createRoot(container).render(
  <React.StrictMode>
    <App client={createHttpDataClient()} />
  </React.StrictMode>,
);
