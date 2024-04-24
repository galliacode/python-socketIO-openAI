import ConnectionInterface from "../interfaces/ConnectionInterface";
import CredentialInterface from "../interfaces/CredentialInterface";
import { get, postURLEncoded } from "../utils/APIHelper";

export const getUserCredential = async (): Promise<CredentialInterface> => {
  const response = await get(
    `${import.meta.env.VITE_SERVER_ADRESS}/conn-params/user_db_creds/all`,
  );
  if (response.ok) {
    return response.response[0];
  }
  throw response.message;
};

export const setUserCredential = async (
  data: CredentialInterface,
): Promise<CredentialInterface> => {
  const response = await postURLEncoded(
    `${import.meta.env.VITE_SERVER_ADRESS}/conn-params/user_db_creds`,
    data,
  );
  if (response.ok) {
    return response.response;
  }
  throw response.message;
};

export const getConnectionTypes = async (): Promise<ConnectionInterface[]> => {
  const response = await get(
    `${import.meta.env.VITE_SERVER_ADRESS}/conn-params/db_conn_types`,
  );
  if (response.ok) {
    return response.response;
  }
  return [];
};

export const testDatabaseConnection = async (
  data: CredentialInterface,
): Promise<boolean | string[]> => {
  const queryString = Object.keys(data)
    .map((key) => key + "=" + encodeURIComponent(data[key]))
    .join("&");
  const response = await get(
    `${
      import.meta.env.VITE_SERVER_ADRESS
    }/conn-params/check_db_connection?${queryString}`,
  );
  if (response.ok) {
    return response.response.schemas;
  }
  throw response.message;
};
