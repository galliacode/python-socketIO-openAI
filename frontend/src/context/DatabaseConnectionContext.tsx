import {
  Alert,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  IconButton,
} from "@mui/material";
import React, { useEffect, useRef } from "react";
import styled from "@emotion/styled";
import TextField from "../components/TextField/TextField";
import Button from "../components/Button/Button";
import { Controller, useForm } from "react-hook-form";
import { useCredentials } from "../hooks/useCredentials";
import { toast } from "react-toastify";
import CustomSelect from "../components/CustomSelect/CustomSelect";
import { useConnectionTypes } from "../hooks/useConnectionTypes";
import CredentialInterface from "../interfaces/CredentialInterface";
import { Close } from "@mui/icons-material";

const StyledClosedButton = styled(IconButton)`
  position: absolute;
  right: 5px;
  top: 5px;
`;

export const DatabaseConnectionContext = React.createContext<{
  databaseConnection: any;
  openDatabaseConnectionModal: () => void;
} | null>(null);
export const DatabaseConnectionProvider = ({
  children,
}: {
  children: React.ReactNode;
}) => {
  const {
    isConnectionTypesLoading,
    connectionTypeMapRef,
    dropdownConnectionTypes,
  } = useConnectionTypes();
  const {
    userCredential,
    isCredentialLoading,
    errorFetchingCredentials,
    setCredential,
    testConnection,
    schemas,
    connectionTested,
    isTestingConnection,
    onCredentialChange,
    isSettingCredential,
  } = useCredentials(isConnectionTypesLoading);
  const isConnectionLoading = isConnectionTypesLoading || isCredentialLoading;
  const {
    register,
    handleSubmit,
    control,
    formState: { errors, dirtyFields },
    reset,
    watch,
  } = useForm({
    defaultValues: {
      name: "",
      conn_type_id: "",
      host: "",
      port: "",
      database_name: "",
      username: "",
      password: "",
      schema_name: "public",
      conn_uuid: undefined || "",
    },
  });

  const watchAllFields = watch();

  const validCredentials = useRef<CredentialInterface>(null);

  const resetFormToUserCredential = () => {
    resetFormCredentials(userCredential, true);
  };
  const resetFormCredentials: (
    credentials: CredentialInterface,
    ignorePassword?: boolean,
  ) => void = (credetials, ignorePassword = false) => {
    reset({
      name: credetials?.conn_name,
      conn_type_id: credetials?.conn_type_id.toString(),
      host: credetials?.host,
      port: credetials?.port.toString(),
      database_name: credetials?.db_name,
      username: credetials?.username,
      password: ignorePassword ? "" : credetials?.password,
      schema_name: credetials?.schema_name || "public",
      conn_uuid: credetials?.conn_uuid,
    });
  };

  useEffect(() => {
    const credentials = {
      name: userCredential?.conn_name,
      conn_uuid: userCredential?.conn_uuid,
      conn_type_id: userCredential?.conn_type_id.toString(),
      host: userCredential?.host,
      port: userCredential?.port.toString(),
      database_name: userCredential?.db_name,
      username: userCredential?.username,
      password: "",
      schema_name: userCredential?.schema_name || "public",
    };
    reset(credentials);
  }, [userCredential]);

  const [databaseConnection] = React.useState(null);
  const [open, setOpen] = React.useState(false);

  const openDatabaseConnectionModal = () => {
    setOpen(true);
    if (userCredential) {
      resetFormToUserCredential();
    }
  };
  const handleClose = (event?, reason?) => {
    if (!userCredential) {
      if (reason === "backdropClick") return;
    }
    setOpen(false);
    onCredentialChange();
    resetFormToUserCredential();
  };

  const onSubmit = async (data: {
    conn_type_id: string;
    host: string;
    port: string;
    database_name: string;
    username: string;
    password: string;
    schema_name: string;
    name: string;
    conn_uuid: string;
  }) => {
    const driver =
      connectionTypeMapRef.current[parseInt(data.conn_type_id)].conn_driver;
    const dialect =
      connectionTypeMapRef.current[parseInt(data.conn_type_id)].conn_dialect;
    const credetials: CredentialInterface = {
      conn_name: data.name,
      conn_uuid: data.conn_uuid,
      conn_type_id: parseInt(data.conn_type_id),
      dialect: dialect,
      host: data.host,
      port: parseInt(data.port),
      username: data.username,
      password: data.password,
      driver: driver,
      db_name: data.database_name,
      schema_name: data.schema_name,
    };
    const response = await testConnection(credetials);
    if (response) {
      validCredentials.current = credetials;
      resetFormCredentials(credetials);
    } else {
      validCredentials.current = null;
      toast.error("Error updating user credential");
    }
  };

  const onSaveCredentials = async () => {
    if (validCredentials.current) {
      const response = await setCredential(validCredentials.current);
      if (response) {
        handleClose();
      }
    }
  };
  const value = { databaseConnection, openDatabaseConnectionModal };

  useEffect(() => {
    if (isConnectionLoading) return;
    if (!userCredential || errorFetchingCredentials) {
      openDatabaseConnectionModal();
    }
  }, [isCredentialLoading]);
  useEffect(() => {
    const isDirty = Object.keys(dirtyFields).length > 0;
    if (isDirty) {
      onCredentialChange();
    }
  }, [watchAllFields]);

  if (isConnectionLoading)
    return (
      <div
        style={{
          height: "100vh",
          width: "100vw",
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
        }}
      >
        <CircularProgress />
      </div>
    );
  return (
    <>
      <Dialog
        open={open}
        onClose={handleClose}
        disableEscapeKeyDown={true}
        slotProps={{
          backdrop: {
            sx: {
              pointerEvents: "none",
            },
          },
        }}
      >
        <form onSubmit={handleSubmit(onSubmit)}>
          <DialogTitle>Connect database</DialogTitle>
          {userCredential && (
            <StyledClosedButton onClick={handleClose}>
              <Close />
            </StyledClosedButton>
          )}
          <DialogContent>
            <form onSubmit={handleSubmit(onSubmit)}>
              <input type="hidden" {...register("conn_uuid")} />
              <TextField
                fullWidth
                label="Connection name"
                helperText={
                  errors.name ? (errors.name.message as string) : null
                }
                error={errors.name ? true : false}
                {...register("name", {
                  required: {
                    message: "name is required",
                    value: true,
                  },
                })}
              />
              <Controller
                control={control}
                name="conn_type_id"
                render={({ field: { onChange, value } }) => (
                  <CustomSelect
                    options={dropdownConnectionTypes}
                    label="Database"
                    value={value}
                    onChange={(event) => {
                      onChange(event.target.value);
                    }}
                  />
                )}
              />
              <TextField
                fullWidth
                label="Host"
                helperText={
                  errors.host ? (errors.host.message as string) : null
                }
                error={errors.host ? true : false}
                {...register("host", {
                  required: {
                    message: "host is required",
                    value: true,
                  },
                })}
              />
              <TextField
                fullWidth
                label="Port"
                helperText={
                  errors.port ? (errors.port.message as string) : null
                }
                error={errors.port ? true : false}
                {...register("port", {
                  required: {
                    message: "port is required",
                    value: true,
                  },
                })}
              />
              <TextField
                fullWidth
                label="Database name"
                helperText={
                  errors.database_name
                    ? (errors.database_name.message as string)
                    : null
                }
                error={errors.database_name ? true : false}
                {...register("database_name", {
                  required: {
                    message: "Database name is required",
                    value: true,
                  },
                })}
              />
              <TextField
                fullWidth
                label="Username"
                helperText={
                  errors.username ? (errors.username.message as string) : null
                }
                error={errors.username ? true : false}
                {...register("username", {
                  required: {
                    message: "username is required",
                    value: true,
                  },
                })}
              />
              <TextField
                fullWidth
                placeholder={
                  userCredential
                    ? "Please re-enter your password to verify changes"
                    : null
                }
                label="Password"
                type="password"
                helperText={
                  errors.password ? (errors.password.message as string) : null
                }
                error={errors.password ? true : false}
                {...register("password", {
                  required: {
                    message: "password is required",
                    value: true,
                  },
                })}
              />
              <Controller
                control={control}
                name="schema_name"
                render={({ field: { onChange, value } }) => (
                  <CustomSelect
                    disabled={!connectionTested}
                    options={schemas.map((schema) => ({
                      label: schema,
                      value: schema,
                    }))}
                    label="Schema"
                    value={value}
                    onChange={(event) => {
                      onChange(event.target.value);
                    }}
                  />
                )}
              />
              {connectionTested && (
                <Alert sx={{ my: 1 }} severity="success">
                  Connection is okay! You can now update your schema.
                </Alert>
              )}
            </form>
          </DialogContent>
          <DialogActions>
            <Button
              type="submit"
              variant="outlined"
              disabled={isTestingConnection}
            >
              {isTestingConnection ? "Testing..." : "Test connection"}
            </Button>
            <Button
              onClick={onSaveCredentials}
              type="button"
              disabled={!connectionTested || isSettingCredential}
            >
              {isSettingCredential ? "Saving..." : "Save"}
            </Button>
          </DialogActions>
        </form>
      </Dialog>
      <DatabaseConnectionContext.Provider value={value}>
        {children}
      </DatabaseConnectionContext.Provider>
    </>
  );
};
