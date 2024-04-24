import React from "react";
import ReactDOM from "react-dom/client";
import { Global } from "@emotion/react";
import "@fontsource/roboto/300.css";
import "@fontsource/roboto/400.css";
import "@fontsource/roboto/500.css";
import "@fontsource/roboto/700.css";
import "@glideapps/glide-data-grid/dist/index.css";
import App from "./App.tsx";
import globalStyles from "./Theme/globalStyles.ts";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <Global styles={globalStyles} />
    <App />
  </React.StrictMode>,
);
