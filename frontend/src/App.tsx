import { ThemeProvider } from "@mui/material";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import { ToastContainer } from "react-toastify";
import { theme } from "./Theme/theme";
import Root from "./routes/Root/root";
import ErrorPage from "./error-page";
import LoginPage from "./routes/Login/login";
import SignupPage from "./routes/Signup/signup";
import ForgotPasswordPage from "./routes/ForgotPassword/forgotPassword";
import ResetPasswordPage from "./routes/ResetPassword/resetPassword";
import ChatPage from "./routes/Chat/chat";
import { RequireAuth } from "./modules/RequireAuth/RequireAuth";
import { AuthProvider } from "./context/AuthContext";
import { DatabaseConnectionProvider } from "./context/DatabaseConnectionContext";
import "react-toastify/dist/ReactToastify.css";

const queryClient = new QueryClient();

const router = createBrowserRouter([
  {
    path: "/",
    element: (
      <RequireAuth>
        <DatabaseConnectionProvider>
          <Root />
        </DatabaseConnectionProvider>
      </RequireAuth>
    ),
    errorElement: <ErrorPage />,
    children: [
      {
        path: "/chat",
        element: <ChatPage />,
        children: [
          {
            path: ":chatId",
            element: <ChatPage />,
          },
        ],
      },
    ],
  },
  {
    path: "/login",
    element: <LoginPage />,
    errorElement: <ErrorPage />,
  },
  {
    path: "/signup",
    element: <SignupPage />,
    errorElement: <ErrorPage />,
  },
  {
    path: "/forgot-password",
    element: <ForgotPasswordPage />,
    errorElement: <ErrorPage />,
  },
  {
    path: "/reset-password",
    element: <ResetPasswordPage />,
    errorElement: <ErrorPage />,
  },
]);

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={theme}>
        <AuthProvider>
          <RouterProvider router={router} />
        </AuthProvider>
      </ThemeProvider>
      <ToastContainer />
    </QueryClientProvider>
  );
}

export default App;
