import { useQuery } from "@tanstack/react-query";
import { getConnectionTypes } from "../services/credentialsService";
import { useRef } from "react";
import ConnectionInterface from "../interfaces/ConnectionInterface";

export const useConnectionTypes = () => {
  const {
    data = [],
    isLoading,
    error,
  } = useQuery({
    queryKey: ["connectionTypes"],
    queryFn: getConnectionTypes,
    retry: false,
  });
  const connectionTypeMapRef = useRef<{
    [key: number]: ConnectionInterface;
  }>({});
  const dropdownConnectionTypes = data.map((connectionType) => {
    connectionTypeMapRef.current[connectionType.id] = connectionType;
    return {
      value: connectionType.id.toString(),
      label: connectionType.conn_type,
    };
  });
  return {
    connectionTypes: data,
    dropdownConnectionTypes,
    isConnectionTypesLoading: isLoading,
    errorFetchingConnectionTypes: error,
    connectionTypeMapRef,
  };
};
