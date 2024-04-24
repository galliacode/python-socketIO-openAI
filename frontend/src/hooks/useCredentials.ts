import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  getUserCredential,
  setUserCredential,
  testDatabaseConnection,
} from "../services/credentialsService";
import { useState } from "react";
import { toast } from "react-toastify";

export const useCredentials = (isConnectionTypesLoading: boolean) => {
  const queryClient = useQueryClient();
  const [schemas, setSchemas] = useState(["public"]);
  const [connectionTested, setConnectionTested] = useState(false);

  const onCredentialChange = () => {
    setConnectionTested(false);
  };

  const { data, isLoading, error } = useQuery({
    queryKey: ["credentials"],
    queryFn: getUserCredential,
    retry: false,
    enabled: !isConnectionTypesLoading,
  });

  const { mutateAsync: setCredential, isPending } = useMutation({
    mutationFn: setUserCredential,
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ["credentials"] });
    },
    onSuccess: () => {
      toast.success("Credentials saved successfully");
    },
    onError: (error: string) => {
      toast.error(error);
    },
  });

  const { mutateAsync: testConnection, isPending: isTestingConnection } =
    useMutation({
      mutationFn: testDatabaseConnection,
      onSuccess: (response) => {
        if (response && typeof response === "object" && response) {
          setConnectionTested(true);
          setSchemas(response);
        }
      },
      onError: (error: string) => {
        toast.error(error);
      },
    });

  return {
    userCredential: data,
    isCredentialLoading: isLoading,
    errorFetchingCredentials: error,
    setCredential,
    testConnection,
    isSettingCredential: isPending,
    schemas,
    connectionTested,
    isTestingConnection,
    onCredentialChange,
  };
};
