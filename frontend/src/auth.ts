import { post, postURLEncoded } from "./utils/APIHelper";

/**
 * This represents some generic auth provider API, like Firebase.
 */

const authProvider = {
  async signin(
    data: {
      username: string;
      password: string;
    },
    callback?: any,
  ) {
    const response = await postURLEncoded(
      `${import.meta.env.VITE_SERVER_ADRESS}/auth/login`,
      data,
      false,
    );
    if (response.ok) {
      localStorage.setItem("token", response.response.access_token);
      if (callback) callback(response);
    } else {
      if (callback) callback(response);
    }
  },
  async sendPasswordReset(email: string, callback?: any) {
    const response = await post(
      `${
        import.meta.env.VITE_SERVER_ADRESS
      }/auth/request-password-reset?email=${encodeURIComponent(email)}`,
      null,
      false,
    );
    if (callback) callback(response);
  },
  async resetPassword(
    data: { token: string; password: string },
    callback?: any,
  ) {
    const response = await postURLEncoded(
      `${import.meta.env.VITE_SERVER_ADRESS}/auth/reset-password`,
      data,
      false,
    );
    if (callback) callback(response);
  },
  signout(callback?: VoidFunction) {
    localStorage.removeItem("token");
    if (callback) callback();
  },
};

export { authProvider };
