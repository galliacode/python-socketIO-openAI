import React from "react";
import { DatabaseConnectionContext } from "../context/DatabaseConnectionContext";

export const useDatabaseConnection = () => {
  return React.useContext(DatabaseConnectionContext);
};
