import { Link, useNavigate } from "react-router-dom";
import Button from "../../components/Button/Button";
import TextField from "../../components/TextField/TextField";
import {
  StyledInputContainer,
  StyledLinkHelper,
  StyledLoginCard,
  StyledLoginCardTitle,
  StyledLoginContainer,
} from "./style";
import { useForm } from "react-hook-form";
import { useAuth } from "../../hooks/useAuth";
import { toast } from "react-toastify";

const LoginPage = () => {
  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm();
  const { signin } = useAuth();
  const navigate = useNavigate();
  const submitForm = async (data: any) => {
    signin(data, (response) => {
      if (response.ok) {
        navigate("/chat");
      } else {
        toast.error(response.message);
      }
    });
  };
  return (
    <StyledLoginContainer>
      <StyledLoginCard>
        <form onSubmit={handleSubmit(submitForm)}>
          <StyledLoginCardTitle>Login</StyledLoginCardTitle>
          <StyledInputContainer>
            <TextField
              fullWidth
              label="Username"
              placeholder="Enter your username"
              helperText={
                errors.username ? (errors.username.message as string) : null
              }
              error={errors.username ? true : false}
              {...register("username", {
                required: {
                  message: "Username is required",
                  value: true,
                },
              })}
            />
          </StyledInputContainer>
          <StyledInputContainer>
            <TextField
              fullWidth
              label="Password"
              type="password"
              placeholder="Enter your password"
              helperText={
                errors.password
                  ? (errors.password.message as string)
                  : "Minimum 8 characters"
              }
              error={errors.password ? true : false}
              {...register("password", {
                required: {
                  message: "Password is required",
                  value: true,
                },
                minLength: {
                  message: "Minimum 8 characters",
                  value: 8,
                },
              })}
            />
          </StyledInputContainer>
          <Button type="submit" fullWidth>
            Login
          </Button>
          <StyledLinkHelper>
            <Link to={"/forgot-password"}>Forgot password?</Link>
          </StyledLinkHelper>
          <StyledLinkHelper>
            Don't have account? <Link to={"/signup"}>Register here!</Link>
          </StyledLinkHelper>
        </form>
      </StyledLoginCard>
    </StyledLoginContainer>
  );
};

export default LoginPage;
