import { Link, useNavigate } from "react-router-dom";
import Button from "../../components/Button/Button";
import TextField from "../../components/TextField/TextField";
import {
  StyledInputContainer,
  StyledLinkHelper,
  StyledSignupContainer,
  StyledSignupCard,
  StyledSignupLoginCardTitle,
} from "./style";
import { useForm } from "react-hook-form";
import { postURLEncoded } from "../../utils/APIHelper";
import { toast } from "react-toastify";

const SignupPage = () => {
  const navigate = useNavigate();
  const {
    register,
    handleSubmit,
    watch,
    formState: { errors },
  } = useForm();
  const onSubmit = async (data: any) => {
    const { username, email, password } = data;
    const requestBody = {
      username,
      email,
      password,
    };
    const response = await postURLEncoded(
      `${import.meta.env.VITE_SERVER_ADRESS}/auth/signup`,
      requestBody,
      false,
    );
    if (response.ok) {
      toast.success("Account created! Please login to continue");
      navigate("/login");
    } else {
      toast.error(response.message);
    }
  };
  return (
    <StyledSignupContainer>
      <StyledSignupCard>
        <form onSubmit={handleSubmit(onSubmit)}>
          <StyledSignupLoginCardTitle>Signup</StyledSignupLoginCardTitle>
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
              label="Email"
              placeholder="Enter your email"
              helperText={
                errors.email ? (errors.email.message as string) : null
              }
              error={errors.username ? true : false}
              {...register("email", {
                required: {
                  message: "email is required",
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
                errors.password ? (errors.password.message as string) : null
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
          <StyledInputContainer>
            <TextField
              fullWidth
              label="Confirm Password"
              type="password"
              placeholder="Confirm your password"
              helperText={
                errors.confirmPassword
                  ? (errors.confirmPassword.message as string)
                  : null
              }
              error={errors.confirmPassword ? true : false}
              {...register("confirmPassword", {
                required: {
                  message: "Password is required",
                  value: true,
                },
                validate: (val) => {
                  if (watch("password") != val) {
                    return "Passwords do not match";
                  }
                },
              })}
            />
          </StyledInputContainer>
          <Button type="submit" fullWidth>
            Register
          </Button>
          <StyledLinkHelper>
            Already have an account? <Link to={"/login"}>Login here!</Link>
          </StyledLinkHelper>
        </form>
      </StyledSignupCard>
    </StyledSignupContainer>
  );
};

export default SignupPage;
