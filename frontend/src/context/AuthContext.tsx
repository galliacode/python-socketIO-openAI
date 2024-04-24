import React from "react";
import { authProvider } from "../auth";
import { jwtDecode } from "jwt-decode";
import { useQueryClient } from "@tanstack/react-query";

export const AuthContext = React.createContext<{
  user: any;
  signin: (newUser: string, callback: any) => void;
  signout: (callback: any) => void;
  3;
  sendPasswordReset: (email: string, callback: any) => void;
  resetPassword: (
    data: { token: string; password: string },
    callback: any,
  ) => void;
}>(null!);

export const AuthProvider = ({ children }: { children: React.ReactNode }) => {
  const [user, setUser] = React.useState("");
  const logoutTimerRef = React.useRef(null);
  const queryClient = useQueryClient();
  const setAutoLogoutTimer = (expTime: number) => {
    const currentTime = Math.floor(Date.now() / 1000); // Current time in seconds
    const expiresIn = (expTime - currentTime) * 1000;
    if (logoutTimerRef.current) {
      clearTimeout(logoutTimerRef.current);
      logoutTimerRef.current = null;
    }
    logoutTimerRef.current = setTimeout(() => {
      authProvider.signout();
      window.location.reload();
    }, expiresIn);
  };
  const isAuthenticated = () => {
    const token = localStorage.getItem("token");
    if (token) {
      const decoded = jwtDecode(token);
      if (decoded?.exp && decoded.exp > Date.now() / 1000) {
        setAutoLogoutTimer(decoded.exp);
        setUser(decoded.sub || "");
        return true;
      }
    }
    return false;
  };
  const signin = (
    data: {
      username: string;
      password: string;
    },
    callback: any,
  ) => {
    return authProvider.signin(data, (response) => {
      callback(response);
    });
  };

  const signout = (callback: VoidFunction) => {
    return authProvider.signout(() => {
      queryClient.clear();
      callback();
    });
  };

  const sendPasswordReset = (email: string, callback: VoidFunction) => {
    return authProvider.sendPasswordReset(email, callback);
  };

  const resetPassword = (
    data: { token: string; password: string },
    callback: VoidFunction,
  ) => {
    return authProvider.resetPassword(data, callback);
  };

  const value: any = {
    isAuthenticated,
    signin,
    signout,
    user,
    sendPasswordReset,
    resetPassword,
  };
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};
